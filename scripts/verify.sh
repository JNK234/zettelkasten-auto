#!/usr/bin/env bash
# Run every check of vaultengine + the vault CLI + vault-tui, and stop at the first failure.
#
#   scripts/verify.sh            everything (about 8 minutes)
#   scripts/verify.sh --quick    skip the two long test suites (pytest, npm test): CLI, API and app end to end only
#
# Needs the installs from docs/install.md: `uv sync` and `cd tui && npm ci`. What each step proves: docs/verify.md.
# Works on scratch copies of the sample vault; nothing in the repo is modified.
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0
[[ ${1:-} == --quick ]] && QUICK=1

PY=.venv/bin/python
VAULT=.venv/bin/vault
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
sample="$scratch/sample-vault"
D="50_MOC/discourse/rl-environment-design"
EVD="EVD - Reward is a deterministic outcome check 1 only if the final database equals the data"
CLM="CLM - Grade the observed end state rather than the trajectory"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() {
	printf '\033[31mFAIL\033[0m %s\n' "$*"
	exit 1
}
ok() { printf '  ok  %s\n' "$*"; }
# expect "<output>" "<text that must appear>"...
expect() {
	local output=$1
	shift
	for wanted in "$@"; do
		grep -qF -- "$wanted" <<<"$output" || fail "expected to see: $wanted"$'\n'"--- got:"$'\n'"$output"
		ok "$wanted"
	done
}
# reject "<output>" "<text that must not appear>"...
reject() {
	local output=$1
	shift
	for unwanted in "$@"; do
		! grep -qF -- "$unwanted" <<<"$output" || fail "did not expect: $unwanted"$'\n'"--- got:"$'\n'"$output"
		ok "not: $unwanted"
	done
}
# run_status <expected exit code> command...: runs it, checks the exit code, leaves the output in $out
run_status() {
	local want=$1
	shift
	set +e
	out=$("$@" 2>&1)
	local got=$?
	set -e
	[[ $got == "$want" ]] || fail "expected exit $want, got $got from: $*"$'\n'"$out"
	ok "exit $got from vault ${2:-$1}${3:+ …}"
}
# A hash of every file in a folder (names, bytes), to prove that something changed nothing.
snapshot() { (cd "$1" && find . -type f ! -path './.vaultengine/index.sqlite*' -print0 | sort -z | xargs -0 sha256sum | sha256sum); }
unchanged() {
	[[ $(snapshot "$1") == "$2" ]] || fail "$3 changed files"
	ok "$3 left every file unchanged"
}

step "Preflight: the installs from docs/install.md"
[[ -x $PY ]] || fail "no .venv: run \`uv sync\` in the repo root"
[[ -x $VAULT ]] || fail "no vault command in .venv: run \`uv sync\`"
$PY -c "import markdownify" 2>/dev/null || fail "markdownify missing: run \`uv sync\`"
[[ -d tui/node_modules ]] || fail "no tui/node_modules: run \`cd tui && npm ci\`"
[[ -f tui/node_modules/node-pty/build/Release/pty.node || -d tui/node_modules/node-pty/prebuilds ]] ||
	fail "node-pty is not built: \`cd tui && npm ci\` (offline: \`npm_config_nodedir=/usr npm ci\`, see docs/install.md)"
node -e 'process.exit(+process.versions.node.split(".")[0] >= 22 ? 0 : 1)' || fail "Node.js 22 or newer is needed"
ok "python $($PY -c 'import sys; print(sys.version.split()[0])'), node $(node --version), node-pty built"

step "Documentation: every link in README.md and docs/ resolves"
expect "$($PY scripts/check_docs.py)" "docs linked from README.md: all resolve"

if [[ $QUICK == 0 ]]; then
	step "Python: engine, index, formats, writes, ingest, JSON-RPC and legacy tests"
	$PY -m pytest -q
fi

step "Index, stats and health (vault index, vault stats, vault health)"
cp -Rp tests/fixtures/vault "$sample"
expect "$($VAULT index "$sample")" "180 notes, 0 attachments" "parsed     180  (180 added, 0 changed, 0 deleted)" "links      1787 resolved this run"
expect "$($VAULT index "$sample")" "parsed     0  (index already up to date)"
echo "Edited." >>"$sample/20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
expect "$($VAULT index "$sample")" "parsed     1  (0 added, 1 changed, 0 deleted)" "~ 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
expect "$($VAULT stats "$sample")" "notes             180" "resolved        472   unresolved 1315 (612 distinct targets) · ambiguous 0"
expect "$($VAULT health "$sample")" "unresolved links                      1315   across 612 missing targets" \
	"orphans (no backlinks)                  41" "dead ends (no outgoing note links)      21" "empty notes                              0"
expect "$(COLUMNS=100 $VAULT search "$sample" "trajectory synthesis" --limit 3)" \
	"  62 of 180 notes match · exact matches first, then score" \
	"   1. single-call-«trajectory»-«synthesis»" "      trajectory: title, heading, body ×2 · synthesis: title, heading"
expect "$(COLUMNS=100 $VAULT search "$sample" "zettel" --limit 3)" "  ── exact ──" "  ── close spellings only ──" \
	"      zettel ≈ zettels (1 edit): heading"
expect "$(COLUMNS=100 $VAULT search "$sample" '"reward hacking"' --limit 1)" '  "reward hacking"  46 notes (phrase, required)'
expect "$(COLUMNS=100 $VAULT search "$sample" 'reward NEAR/3 model' --limit 1)" "  reward NEAR/3 model  31 notes (near, required)"
expect "$($VAULT search "$sample" 'NEAR/3 model' 2>&1 || true)" "error: NEAR/3 needs a single word on each side"
expect "$(COLUMNS=100 $VAULT search "$sample" '"reward hacking" links-to:[[grpo-group-relative-policy-optimization]] -path:10_SOURCES')" \
	"  2 of 180 notes match · by score" "   1. instance-specific-rubric-as-reward-function"
expect "$(COLUMNS=100 $VAULT search "$sample" 'within:2 of:[[obfuscated-reward-hacking]] tag:ai-safety')" "  12 of 180 notes match" "     12  #ai-safety"
expect "$(COLUMNS=100 $VAULT search "$sample" 'missing:status')" "  3 of 180 notes match" "      2  50_MOC/topics/"
expect "$($VAULT search "$sample" 'reward (tag:x OR b' 2>&1 || true)" "error: this ( is never closed" "         ^"
expect "$(COLUMNS=100 $VAULT related "$sample" kl-penalty-in-rlhf --limit 3 --color never)" "   2. RLHF" \
	"      2 hops via direct-preference-optimization-dpo +1 · similar text (kl, dpo, rlhf)"
expect "$(COLUMNS=100 $VAULT suggest "$sample" --limit 3 --color never)" "    1. rl-environment-design" \
	"    [[on-policy-vs-off-policy-learning]] (5 notes) → 20_ZETTLEKASTEN/on-policy-versus-off-policy-learning.md"
expect "$($VAULT suggest "$sample" --draft reward-model-overoptimization-hump)" "# reward-model-overoptimization-hump" "## What the linking notes say"
expect "$(COLUMNS=120 $VAULT suggest "$sample" --links --path 20_ZETTLEKASTEN --limit 3 --color never)" \
	"in-context-scheming-capability  ↔  sandbagging-strategic-underperformance" "cited together by scheming-as-threat-model"
expect "$(COLUMNS=100 $VAULT unlinked "$sample" monitorability-tax --color never)" \
	"  4 plain-text mentions of monitorability-tax in 1 note that does not link to it yet"

step "JSON-RPC 2.0 over stdio (vault serve)"
rpc=$(printf '%s\n' \
	'{"jsonrpc":"2.0","id":1,"method":"notes.pick","params":{"query":"klpen","limit":1}}' \
	'{"jsonrpc":"2.0","id":2,"method":"notes.delete"}' \
	'{"jsonrpc":"1.0","id":3,"method":"index.status"}' |
	$VAULT serve "$sample" 2>/dev/null)
expect "$rpc" '"path":"20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"' '"id":2,"error":{"code":-32601' '"id":3,"error":{"code":-32600'
rpc=$(printf '%s\n' \
	'{"jsonrpc":"2.0","id":10,"method":"search.query","params":{"query":"reward hacking tag:ai-safety","limit":1}}' \
	'{"jsonrpc":"2.0","id":11,"method":"search.query","params":{"query":"reward (tag:ai-safety OR path:50_MOC"}}' \
	'{"jsonrpc":"2.0","id":12,"method":"note.preview","params":{"path":"20_ZETTLEKASTEN/monitorability-tax.md","query":"monitorability"}}' \
	'{"jsonrpc":"2.0","id":13,"method":"note.related","params":{"note":"kl-penalty-in-rlhf","limit":2}}' \
	'{"jsonrpc":"2.0","id":14,"method":"links.suggest","params":{"limit":1,"path":"20_ZETTLEKASTEN"}}' \
	'{"jsonrpc":"2.0","id":15,"method":"notes.suggest","params":{"limit":1}}' \
	'{"jsonrpc":"2.0","id":16,"method":"mentions.unlinked","params":{"note":"monitorability-tax"}}' \
	'{"jsonrpc":"2.0","id":17,"method":"note.related","params":{"note":"no-such-note"}}' |
	$VAULT serve "$sample" 2>/dev/null)
expect "$rpc" '"id":10,"result":{"query":"reward hacking tag:ai-safety","tree":"(and (tag ai-safety) (any reward hacking))","notes":180,"total":9' \
	'"id":11,"error":{"code":-32006,"message":"this ( is never closed","data":{"query":"reward (tag:ai-safety OR path:50_MOC","start":7,"end":8,"hint":"add ) after the group"}}' \
	'"id":12,"result":{"path":"20_ZETTLEKASTEN/monitorability-tax.md","title":"monitorability-tax","query":"monitorability","matched":true' \
	'"title":"RLHF","score":' \
	'"id":14,"result":{"note":null,"path":"20_ZETTLEKASTEN","links":[{"a":"20_ZETTLEKASTEN/in-context-scheming-capability.md","b":"20_ZETTLEKASTEN/sandbagging-strategic-underperformance.md"' \
	'"id":15,"result":{"total_missing":612,"notes":[{"name":"rl-environment-design","linking_notes":52' \
	'"id":16,"result":{"note":"20_ZETTLEKASTEN/monitorability-tax.md","total":4,"notes":1' \
	'"id":17,"error":{"code":-32002,"message":"no note or link target called \"no-such-note\""'

step "The format file (vault formats)"
expect "$($VAULT formats "$sample" --format brain)" "180 typed, 0 untyped" "issues     102   in 77 notes" \
	"  zettel           109     67      42   in 20_ZETTLEKASTEN/" "  missing-heading          28 / 28" "     481  tension"
bad="$scratch/bad.yaml"
sed 's/required: true, kind: tags, multiple: false, values: \[baby/require: true, kind: tags, multiple: false, values: [baby/' formats/brain.yaml >"$bad"
run_status 2 $VAULT check "$sample" --format "$bad"
expect "$out" "types.zettel.metadata.Status: unknown key 'require'"

step "Check, and its exit codes (vault check, vault check --changed)"
run_status 1 $VAULT check "$sample" --format brain
expect "$out" "  checked 180 notes" "✗ 102 violations in 77 notes" "      59  unresolved-target" \
	"    138  missing-heading      missing heading 'Notes' (expected at the end of the note)"
noindex="$scratch/no-index"
cp -Rp tests/fixtures/vault "$noindex"
run_status 2 $VAULT check "$noindex" --format brain --changed
expect "$out" "--changed needs an index"
run_status 0 $VAULT check "$sample" --format brain --changed
expect "$out" "✓ no violations in 0 notes checked"
claim="$sample/$D/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md"
sed -i.bak -e '/^supported-by::/d' -e 's/^Status: #supported/Status: #suported/' "$claim" && rm "$claim.bak"
run_status 1 $VAULT check "$sample" --format brain --changed
expect "$out" "  checked 1 note changed since the last index" "✗ 3 violations in 1 note" \
	"      5  bad-value            Status: #suported is not an allowed value (supported, contested, weak)" \
	"     10  missing-mirror       missing \`supported-by:: [[EVD - Where there is no real state"
json=$($VAULT check "$sample" --format brain --changed --json || true)
expect "$json" '"ok": false' '"mode": "changed"'

step "Fix: dry run first, then apply (vault fix)"
before=$(snapshot "$sample")
run_status 0 $VAULT fix "$sample" --format brain --dry-run
expect "$out" "+supported-by:: [[EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a]]" \
	"+## What This Is" "Would fix 29 notes:" "Dry run: nothing was written."
unchanged "$sample" "$before" "fix --dry-run"
run_status 0 $VAULT fix "$sample" --format brain
expect "$out" "Fixed 29 notes:" "Violations left after this fix, for you to fix: 59 unresolved-target, 15 missing-key, 1 bad-value"
expect "$(cat "$claim")" "supported-by:: [[EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a]]"
run_status 1 $VAULT check "$sample" --format brain
expect "$out" "✗ 75 violations in 50 notes"
reject "$out" "missing-heading" "missing-mirror"
run_status 0 $VAULT fix "$sample" --format brain
expect "$out" "Nothing to fix automatically."
# The 28 notes fix repaired are indexed again; the hand-edited claim still has a violation, so it stays pending.
run_status 1 $VAULT check "$sample" --format brain --changed
expect "$out" "  checked 1 note changed since the last index" "✗ 1 violation in 1 note"

step "Safe writes: new, link with its mirror, set, and refusals (vault new, vault link, vault set)"
$VAULT index "$sample" >/dev/null
run_status 0 $VAULT new "$sample" -f brain --type zettel --title "Verifier Hacking Needs a Do-Nothing Baseline" \
	--set "Tags=[[evaluation-methodology]]" --section "What This Is" "A grader a do-nothing agent passes pays for inaction."
expect "$out" "+ 20_ZETTLEKASTEN/verifier-hacking-needs-a-do-nothing-baseline.md" "check    no new violations" "index    updated 1 note"
zettel=$(cat "$sample/20_ZETTLEKASTEN/verifier-hacking-needs-a-do-nothing-baseline.md")
expect "$zettel" "Status: #baby" "Tags: [[evaluation-methodology]]" "# Verifier Hacking Needs a Do-Nothing Baseline" \
	$'## What This Is\nA grader a do-nothing agent passes pays for inaction.' "## Tensions" "# References"
run_status 0 $VAULT link "$sample" -f brain "$EVD" "$CLM" --as supports
expect "$out" "Wrote 2 notes (link):" "to \`supports::\`" "to \`supported-by::\`" "index    updated 2 notes"
expect "$(grep '^supports::' "$sample/$D/$EVD.md")" "[[$CLM]]"
expect "$(grep '^supported-by::' "$sample/$D/$CLM.md")" "[[$EVD]]"
run_status 0 $VAULT link "$sample" -f brain "$EVD" "$CLM" --as supports
expect "$out" "already has \`supports:: [[$CLM]]\`" "already has \`supported-by:: [[$EVD]]\`" "Nothing to write."
run_status 0 $VAULT set "$sample" -f brain "20_ZETTLEKASTEN/cot-monitor-beats-action-monitor.md" Created "24-09-2026 16:29"
expect "$out" "before line 1: Created: 24-09-2026 16:29" "index    updated 1 note"
expect "$($VAULT index "$sample")" "parsed     0  (index already up to date)"
before=$(snapshot "$sample")
run_status 1 $VAULT set "$sample" -f brain "$CLM" Status "#settled"
expect "$out" "✗ refused: the change would add 1 violation; nothing was written" "+Status: #settled"
run_status 1 $VAULT link "$sample" -f brain "$EVD" kl-penalty-in-rlhf --as supports
expect "$out" "relation-target-type \`supports::\` [[kl-penalty-in-rlhf]] is a zettel; expected claim"
run_status 1 $VAULT new "$sample" -f brain --type claim --folder "$D" --title "Graders must be tested" --set "Status=#supported"
expect "$out" "missing relation field \`supported-by::\` (required for claim)"
run_status 2 $VAULT new "$sample" -f brain --type zettel --title "KL Penalty in RLHF" --set "Tags=[[x]]"
expect "$out" "a note named 'kl-penalty-in-rlhf' already exists"
run_status 2 $VAULT link "$sample" -f brain "No such note" "$CLM" --as supports
expect "$out" "error: no note 'No such note'"
unchanged "$sample" "$before" "refused and rejected writes"
expect "$($VAULT index "$sample")" "parsed     0  (index already up to date)"

step "The duplicate gate and ingest (vault ingest)"
cp "$sample/10_SOURCES/2411.00640 - Adding Error Bars to Evals.md" "$scratch/error-bars-copy.md"
printf '%s\n' "<html><head><meta property='og:title' content='Adding Error Bars to Evals: A Statistical Approach to" \
	"Language Model Evaluations'></head><body><main><p>A short abstract in other words.</p></main></body></html>" >"$scratch/abs-page.html"
printf '%s\n' '<html><head><title>Reward Model Ensembles | Notes</title></head><body><article>' \
	'<h1>Reward Model Ensembles</h1><p>An ensemble of reward models trained from different seeds makes different errors in' \
	'different places, so the mean is less enthusiastic exactly where any single model is wrong, which delays the point' \
	'where optimizing the proxy stops improving the true objective and starts to hurt it.</p></article></body></html>' \
	>"$scratch/ensembles.html"
printf '%s\n' '<html><head><title>Measuring AI Ability to Complete Short Tasks</title></head><body><article>' \
	'<p>Short tasks of a few minutes are where most benchmarks live, and where models now succeed almost always. This' \
	'note looks at what a suite of sub-minute coding chores still tells us once success rates are near the ceiling.</p>' \
	'</article></body></html>' >"$scratch/short-tasks.html"
before=$(snapshot "$sample")
run_status 1 $VAULT ingest "$sample" "$scratch/error-bars-copy.md" --as source -f brain
expect "$out" "✗ not ingested: error-bars-copy.md duplicates 1 existing note; nothing was written" \
	"      content overlap   1.000  duplicate (threshold 0.8)   7591 of 7591 five-word sequences shared" \
	"      same file         yes    identical SHA-256"
run_status 1 $VAULT ingest "$sample" "$scratch/abs-page.html" --as source -f brain
expect "$out" "      title similarity  1.000  duplicate (threshold 0.9)"
unchanged "$sample" "$before" "the duplicate gate"
run_status 0 $VAULT ingest "$sample" "$scratch/short-tasks.html" --as source -f brain   # "… Long Tasks" exists: 0.857
expect "$out" "+ 10_SOURCES/Measuring AI Ability to Complete Short Tasks.md"
run_status 0 $VAULT ingest "$sample" "$scratch/ensembles.html" --as source -f brain
expect "$out" "+ 10_SOURCES/Reward Model Ensembles.md" "index    updated 1 note"
expect "$(cat "$sample/10_SOURCES/Reward Model Ensembles.md")" "Original: ensembles.html" "Content hash: sha256:" "## References" "# Notes"
run_status 1 $VAULT ingest "$sample" "$scratch/ensembles.html" --as source -f brain
expect "$out" "same file         yes" "title similarity  1.000  duplicate"

step "Coverage (vault coverage)"
run_status 0 $VAULT coverage "$sample" -f brain
expect "$out" "sources           30   23 linked from other notes, 7 that nothing links to" \
	"  10_SOURCES/Reward Model Ensembles.md" "Most cited sources (citing notes)"

step "JSON-RPC format methods (vault serve --format brain)"
fresh="$scratch/rpc-vault"
cp -Rp tests/fixtures/vault "$fresh"
before=$(snapshot "$fresh")
rpc=$(printf '%s\n' \
	'{"jsonrpc":"2.0","id":1,"method":"vault.check","params":{"limit":1}}' \
	'{"jsonrpc":"2.0","id":2,"method":"vault.coverage","params":{"limit":1}}' \
	"{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"note.set\",\"params\":{\"note\":\"$CLM\",\"key\":\"Status\",\"value\":\"#settled\"}}" \
	"{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"note.link\",\"params\":{\"source\":\"No such note\",\"target\":\"$CLM\",\"relation\":\"supports\"}}" \
	'{"jsonrpc":"2.0","id":5,"method":"note.ingest","params":{"content":"x","filename":"a.pdf"}}' |
	$VAULT serve "$fresh" --format brain 2>/dev/null)
expect "$rpc" '"summary":{"violations":102,"notes":77' '"sources":{"total":28,"linked":23' \
	'"id":3,"result":{"operation":"set","status":"refused"' '"id":4,"error":{"code":-32002' '"id":5,"error":{"code":-32004'
unchanged "$fresh" "$before" "refused and rejected writes over JSON-RPC (not even a lock file)"
rpc=$(printf '%s\n' \
	"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"note.link\",\"params\":{\"source\":\"$EVD\",\"target\":\"$CLM\",\"relation\":\"supports\"}}" \
	'{"jsonrpc":"2.0","id":2,"method":"vault.check","params":{"limit":1}}' |
	$VAULT serve "$fresh" --format brain 2>/dev/null)
expect "$rpc" '"id":1,"result":{"operation":"link","status":"written"' '"summary":{"violations":102,"notes":77'
expect "$(grep '^supported-by::' "$fresh/$D/$CLM.md")" "[[$EVD]]"
rpc=$(printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"vault.check"}' | $VAULT serve "$fresh" 2>/dev/null)
expect "$rpc" '"error":{"code":-32003'

if [[ $QUICK == 0 ]]; then
	step "App: generated types, unit, frame and integration tests (npm test)"
	(cd tui && npm test)
fi

step "App: type check"
(cd tui && npm run typecheck)

step "App: end to end in a real pseudo-terminal (/health; /check, /coverage, /link, /check changed)"
(cd tui && npm run test:e2e)

printf '\n\033[32mALL CHECKS PASSED\033[0m\n'
