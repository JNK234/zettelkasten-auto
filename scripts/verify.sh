#!/usr/bin/env bash
# Run every check of vaultengine + the vault CLI + vault-tui, and stop at the first failure.
#
#   scripts/verify.sh            everything (about 7 minutes on an Apple-silicon Mac, up to 25 on a slower machine)
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
[[ -f models/potion-base-8M/model.safetensors && -f models/potion-base-8M/tokenizer.json ]] ||
	fail "no embedding model in models/potion-base-8M (shipped next to the repository; see docs/install.md)"
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
expect "$(COLUMNS=100 $VAULT explain "$sample" RLHF kl-penalty-in-rlhf --color never)" "  text            0.58 ×   0.30 =        0.173" \
	"180 notes had no cached vector: embedded in memory for this run only"
expect "$(COLUMNS=100 $VAULT related "$sample" kl-penalty-in-rlhf --limit 3 --color never)" "   3. RLHF" \
	"linked from it · shares [[policy-gradient-methods]] · similar text (policy, kl, divergence)" \
	"      2 hops via direct-preference-optimization-dpo +1 · similar text (rlhf, reward, policy)" \
	"· 179 notes compared · text: potion-base-8M embeddings"
expect "$(COLUMNS=100 $VAULT related "$sample" kl-penalty-in-rlhf --limit 3 --unlinked --color never)" "   1. RLHF" \
	"score = 0.1 × graph + 0.05 × cocitation + 0.05 × coupling + 0.7 × text + 0.1 × tags"
expect "$(COLUMNS=100 $VAULT suggest "$sample" --limit 3 --color never)" "    1. rl-environment-design" \
	"    [[on-policy-vs-off-policy-learning]] (5 notes) → 20_ZETTLEKASTEN/on-policy-versus-off-policy-learning.md"
expect "$($VAULT suggest "$sample" --draft reward-model-overoptimization-hump)" "# reward-model-overoptimization-hump" "## What the linking notes say"
expect "$(COLUMNS=120 $VAULT suggest "$sample" --links --path 20_ZETTLEKASTEN --limit 3 --color never)" \
	"in-context-scheming-capability  ↔  scheming-inability-safety-case-closed" "cited together by scheming-as-threat-model" \
	"similar text (scheming, context, reasoning)"
expect "$(COLUMNS=100 $VAULT unlinked "$sample" monitorability-tax --color never)" \
	"  4 plain-text mentions of monitorability-tax in 1 note that does not link to it yet"

step "Settings and embeddings (vault config show, vault embed)"
expect "$(VAULTENGINE_FORMAT=brain $VAULT config show "$sample" --setting embedding.model_id=other-id)" \
	"precedence: flag > env > file > default" "env VAULTENGINE_FORMAT" "flag --setting embedding.model_id" \
	"$(pwd)/models/potion-base-8M"
run_status 2 env VAULTENGINE_EMBEDDING__PROVIDER=nope $VAULT config show
expect "$out" "engine.embedding.provider: unknown provider 'nope'; known: model2vec  [from env VAULTENGINE_EMBEDDING__PROVIDER]"
# related, suggest and explain ran above without cached vectors: they computed them in memory and wrote nothing.
expect "$($VAULT embed "$sample" --status)" "vectors    0 of 180 indexed notes · 180 missing"
expect "$($VAULT embed "$sample")" "potion-base-8M (model2vec, 256 dims)" "vectors    180 embedded, 0 cached (180 notes)"
echo "Edited again." >>"$sample/20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
expect "$($VAULT embed "$sample")" "index      1 parsed (0 added, 1 changed, 0 deleted)" "vectors    1 embedded, 179 cached (180 notes)" \
	"    + 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
expect "$($VAULT embed "$sample" --setting embedding.model_id=other-id)" "vectors    180 embedded, 0 cached (180 notes)"
expect "$($VAULT embed "$sample" --status)" "vectors    180 of 180 indexed notes (256 dims) · all up to date" \
	"other ids  other-id: 180 vectors (256 dims)"
# Without the model only the text signal is off: related and explain still answer, from links and tags.
run_status 0 $VAULT related "$sample" kl-penalty-in-rlhf --color never --setting embedding.model_path=/no/such/model
expect "$out" "score = 0.71 × graph + 0.29 × tags" "text signal unavailable: embedding model folder not found: /no/such/model"
run_status 0 $VAULT explain "$sample" kl-penalty-in-rlhf direct-preference-optimization-dpo --color never --setting embedding.model_path=/no/such/model
expect "$out" "  direct links    1.00 ×   0.43 =        0.429" "  text  unavailable  (embedding model folder not found: /no/such/model)"
A="2312.06942 - AI Control - Improving Safety Despite Intentional Subversion"
B="Shlegeris Greenblatt - The Case for Ensuring Powerful AIs are Controlled"
expect "$($VAULT explain "$sample" "$A" "$B" --color never)" "  score 0.66 = the sum of weight × value" \
	"    2412.14093 - Alignment Faking in Large Language Models       0.72  (links to 3)" \
	"    Shlegeris Greenblatt - Catching AIs Red-Handed (no note yet)  0.72  (linked from 3)" \
	"(sum 0.673)"

step "Map and lenses (vault map, vault deepen)"
out=$(COLUMNS=120 $VAULT map "$sample" -f brain --color never --limit 0)
expect "$out" "tag links: format brain (Tags)" "modularity 0.67 · 12 communities hold 179 notes; 1 note in none" \
	"  topic pages  notes the format brain types as topic" "  No topic page  (10 of 12 communities" \
	"      1. 0.106 (p100)  0.048 (p4)    reward-hacking-as-specification-gaming  → 4, 2, 5, 6, 3"
[[ $($VAULT map "$sample" -f brain --json | sha256sum) == $($VAULT map "$sample" -f brain --json | sha256sum) ]] ||
	fail "vault map gave two different answers"
ok "vault map: the same answer twice (seeded)"
out=$(COLUMNS=120 $VAULT deepen "$sample" reward-hacking-as-specification-gaming -f brain --color never)
expect "$out" "  Neighbourhood  (vault related: 18 of 179 notes at or above the 90th percentile" "  Cited together" \
	"  Same foundations" "      1. cot-monitor-beats-action-monitor  0.34 (p98)" \
	"this note is 1 step off it, joining at model-awareness-of-intent-violation-while-cheating" \
	"  Cluster  (community 2: #ai-safety · control-evaluations-red-team-blue-team, 32 notes)"

step "Discover lenses and meaning edges (vault discover, vault map --meaning)"
out=$(COLUMNS=140 $VAULT map "$sample" -f brain --color never --meaning)
expect "$out" "  meaning      29 edges added for the communities: mutual nearest neighbours by embedding" \
	"  What the meaning edges change  (communities 12 → 11; notes in none 1 → 2)"
out=$(COLUMNS=140 $VAULT discover "$sample" reward-hacking-as-specification-gaming -f brain --color never)
expect "$out" "  Hidden link" "      1. reinforcement-learning-in-language-models" "  Same role elsewhere" \
	"  Bridge  (betweenness 0.106, p100; constraint 0.048, p4" "  Wander  (a random walk on links, seed 42" \
	"    with embeddings (merged by embedding of the names"
[[ $($VAULT discover "$sample" kl-penalty-in-rlhf --json | sha256sum) == $($VAULT discover "$sample" kl-penalty-in-rlhf --json | sha256sum) ]] ||
	fail "vault discover gave two different answers"
ok "vault discover: the same answer twice (seed 42)"
out=$(COLUMNS=140 $VAULT discover "$sample" -f brain --color never --lens gap)
expect "$out" "    without them (closeness of key-term profiles" \
	"         join EVD - Verifiers makes an environment a … ↔ environment-as-installable-package-not-… (3.81, via 'package', 'installable', 'intellect')"

step "Experiments: hidden links, seeded (vault experiment)"
out=$($VAULT experiment "$sample" -f brain --seeds 2 --question relatedness --quiet)
expect "$out" "| relatedness | whole vault | MRR |" "| links only |" "Hidden per seed: 55 of 367 linked pairs; 55 scored per seed." \
	"- \`related.unlinked_weights\`: "
[[ $(echo "$out" | sed -E 's/[0-9]+\.[0-9]{2} \|/t |/g; s/Whole run: [0-9]+ s//; /^Time: /d') == \
	$($VAULT experiment "$sample" -f brain --seeds 2 --question relatedness --quiet | sed -E 's/[0-9]+\.[0-9]{2} \|/t |/g; s/Whole run: [0-9]+ s//; /^Time: /d') ]] ||
	fail "vault experiment gave two different reports"
ok "vault experiment: the same numbers twice (seeds 42, 43; timings aside)"
run_status 2 $VAULT experiment "$sample" --seeds 1
expect "$out" "error: --seeds must be 2 or more"

step "Explore, its memory, link it and map history (vault explore, mark, kept, link-it, unlink, map --record/--since)"
notes_only() { (cd "$1" && find . -type f ! -path './.vaultengine/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum); }
before=$(notes_only "$sample")
out=$(COLUMNS=120 $VAULT map "$sample" -f brain --record --color never)
expect "$out" "  recorded     snapshot " "4 gaps (meaning), 612 missing notes"
out=$(COLUMNS=120 $VAULT explore "$sample" -f brain --color never)
expect "$out" "where to start" "  1. Gap  (discover · " "  4. No topic page  (map · " "→ explore [[llm-as-a-judge-framework]]"
out=$(COLUMNS=120 $VAULT explore "$sample" kl-penalty-in-rlhf -f brain --color never)
expect "$out" "equal space: deepen" "  1. Neighbourhood  (deepen · " "      1.1  RLHF" "           neighbourhood · 2 hops via" \
	"  6. Hidden link  (discover · " "      6.1  covariance-targeted-entropy-intervention" "  8. Gap  (discover · "
[[ $(COLUMNS=120 $VAULT explore "$sample" kl-penalty-in-rlhf -f brain --color never | sha256sum) == $(echo "$out" | sha256sum) ]] ||
	fail "vault explore gave two different answers"
ok "vault explore: the same output twice"
[[ -f "$sample/.vaultengine/stats.sqlite" ]] || fail "vault explore kept no vault-wide numbers (.vaultengine/stats.sqlite)"
ok "vault explore kept the vault-wide numbers for the next run (.vaultengine/stats.sqlite)"
expect "$($VAULT mark "$sample" keep 1.1 6.1)" "  1.1  kept  neighbourhood · RLHF"
expect "$($VAULT mark "$sample" dismiss 7.1)" "  7.1  dismissed  role · rubric-agreement-does-not-establish-rubric-validity"
expect "$($VAULT kept "$sample")" "  covariance-targeted-entropy-intervention" "    hidden · no link either way"
out=$($VAULT link-it "$sample" 8.1 -f brain)
expect "$out" "Would write 2 notes (link-it):" "Dry run: nothing was written."
[[ $(notes_only "$sample") == "$before" ]] || fail "explore, mark, kept, map --record or a dry run changed a note"
ok "explore, mark, kept, map --record and the dry run changed no note"
expect "$($VAULT link-it "$sample" 8.1 -f brain --write)" "Wrote 2 notes (link-it):" "  check    no new violations"
expect "$(grep -h -- '— gap: ' "$sample/20_ZETTLEKASTEN/persona-driven-synthetic-data-generation.md")" \
	"- [[reward-computation-as-a-heavyweight-pipeline-stage]] — gap: "
out=$(COLUMNS=120 $VAULT explore "$sample" kl-penalty-in-rlhf -f brain --color never --fresh)
reject "$out" "rubric-agreement-does-not-establish-rubric-validity" "      1.1  RLHF"
ok "--fresh: the dismissed note and what was seen are gone"
out=$(COLUMNS=120 $VAULT map "$sample" -f brain --since "$(date +%F)" --color never)
expect "$out" "(1 of 4 recorded gaps closed" "    closed  communities 6 and 12" \
	"new link persona-driven-synthetic-data-generation ↔ reward-computation-as-a-heavyweight-pipeline…"
expect "$($VAULT unlink "$sample" reward-computation-as-a-heavyweight-pipeline-stage persona-driven-synthetic-data-generation -f brain --write)" \
	"Wrote 2 notes (unlink):"
[[ $(notes_only "$sample") == "$before" ]] || fail "unlink did not restore both notes"
ok "unlink: both notes byte for byte as before"
[[ -n $(tail -c1 "$sample/10_SOURCES/RLHF.md") ]] || fail "10_SOURCES/RLHF.md was expected to have no final newline"
$VAULT link-it "$sample" kl-penalty-in-rlhf RLHF -f brain --write >/dev/null
[[ -n $(tail -c1 "$sample/10_SOURCES/RLHF.md") ]] || fail "link-it added a final newline to 10_SOURCES/RLHF.md"
$VAULT unlink "$sample" kl-penalty-in-rlhf RLHF -f brain --write >/dev/null
[[ $(notes_only "$sample") == "$before" ]] || fail "link-it then unlink on RLHF.md (no final newline) did not restore both notes"
ok "link-it then unlink on a note without a final newline (10_SOURCES/RLHF.md): byte for byte as before"

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
	'{"jsonrpc":"2.0","id":13,"method":"note.related","params":{"note":"kl-penalty-in-rlhf","limit":3}}' \
	'{"jsonrpc":"2.0","id":14,"method":"links.suggest","params":{"limit":1,"path":"20_ZETTLEKASTEN"}}' \
	'{"jsonrpc":"2.0","id":15,"method":"notes.suggest","params":{"limit":1}}' \
	'{"jsonrpc":"2.0","id":16,"method":"mentions.unlinked","params":{"note":"monitorability-tax"}}' \
	'{"jsonrpc":"2.0","id":17,"method":"note.related","params":{"note":"no-such-note"}}' |
	$VAULT serve "$sample" 2>/dev/null)
expect "$rpc" '"id":10,"result":{"query":"reward hacking tag:ai-safety","tree":"(and (tag ai-safety) (any reward hacking))","notes":180,"total":9' \
	'"id":11,"error":{"code":-32006,"message":"this ( is never closed","data":{"query":"reward (tag:ai-safety OR path:50_MOC","start":7,"end":8,"hint":"add ) after the group"}}' \
	'"id":12,"result":{"path":"20_ZETTLEKASTEN/monitorability-tax.md","title":"monitorability-tax","query":"monitorability","matched":true' \
	'"title":"RLHF","score":' \
	'"id":14,"result":{"note":null,"path":"20_ZETTLEKASTEN","links":[{"a":"20_ZETTLEKASTEN/in-context-scheming-capability.md","b":"20_ZETTLEKASTEN/scheming-inability-safety-case-closed.md"' \
	'"id":15,"result":{"total_missing":612,"notes":[{"name":"rl-environment-design","linking_notes":52' \
	'"id":16,"result":{"note":"20_ZETTLEKASTEN/monitorability-tax.md","total":4,"notes":1' \
	'"id":17,"error":{"code":-32002,"message":"no note or link target called \"no-such-note\""'
rpc=$(printf '%s\n' \
	'{"jsonrpc":"2.0","id":20,"method":"explore.run","params":{"note":"kl-penalty-in-rlhf"}}' \
	'{"jsonrpc":"2.0","id":21,"method":"explore.mark","params":{"verdict":"keep","refs":["1.1"]}}' \
	'{"jsonrpc":"2.0","id":22,"method":"explore.link","params":{"ref":"1.1"}}' \
	'{"jsonrpc":"2.0","id":23,"method":"config.show"}' \
	'{"jsonrpc":"2.0","id":24,"method":"explore.run","params":{"lenses":["hidden"]}}' |
	$VAULT serve "$sample" --format brain 2>/dev/null)
expect "$rpc" '"id":20,"result":{"start":"20_ZETTLEKASTEN/kl-penalty-in-rlhf.md","start_kind":"note"' \
	'"id":21,"result":{"marked":[{"ref":"1.1","lens":"neighbourhood","target":"10_SOURCES/RLHF.md"' \
	'"id":22,"result":{"operation":"link-it","status":"dry-run","dry_run":true,"written":false' \
	'"id":23,"result":{"config_file":' '"id":24,"error":{"code":-32007,"message":"hidden only run from a note"'

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

step "The core format, a baseline, the pipeline (vault check -f core --baseline, vault pipeline, vault note)"
core="$scratch/core-vault"
cp -Rp tests/fixtures/vault "$core"
before=$(snapshot "$core")
run_status 1 $VAULT check "$core" -f core
expect "$out" "  checked 180 notes" "✗ 144 violations in 123 notes" "      83  too-few-links" "      43  missing-key"
run_status 2 $VAULT check "$core" -f core --baseline
expect "$out" "error: no baseline for format 'core'" "--update-baseline"
unchanged "$core" "$before" "check and check --baseline without a baseline"
run_status 0 $VAULT check "$core" -f core --update-baseline
expect "$out" "Recorded a baseline of 144 violations: $core/.vaultengine/baseline-core.json"
run_status 0 $VAULT check "$core" -f core --baseline
expect "$out" "✓ no new violations in 180 notes checked (144 known violations in the baseline not shown)"
mv "$core/20_ZETTLEKASTEN/reward.md" "$core/20_ZETTLEKASTEN/reward-signal.md"
run_status 0 $VAULT check "$core" -f core --baseline
expect "$out" "followed a renamed note (same text): 20_ZETTLEKASTEN/reward.md → 20_ZETTLEKASTEN/reward-signal.md"
printf '%s\n' "Created: 09-11-2026 21:20" "Tags: [[rl-environment-design]]" "" "# No Source" "See [[kl-penalty-in-rlhf]]." \
	>"$core/20_ZETTLEKASTEN/no-source.md"
printf '%s\n' "Created: 09-11-2026 21:20" "Tags: [[rl-environment-design]]" "" "# Thin" "See [[no-source]]." \
	>"$core/50_MOC/topics/thin.md"
mkdir "$core/10_SOURCES/old" && printf 'x\n' >"$core/10_SOURCES/old/filed-away.md"
run_status 1 $VAULT check "$core" -f core --baseline
expect "$out" "✗ 3 new violations in 3 notes (144 known violations in the baseline not shown)" \
	"      1  too-few-links        links to 0 source notes; a zettel needs at least 1" \
	"      1  too-few-links        links to 1 zettel note; a page needs at least 3" \
	"      1  untyped-note         no type matches this note, and every note in 10_SOURCES/ needs one"
reject "$out" "missing-key"
run_status 0 $VAULT check "$core" -f core --update-baseline
expect "$out" "Recorded a baseline of 147 violations (it had 144)"
run_status 0 $VAULT pipeline "$core" -f core
expect "$out" "    compiled                3" "    extracted              12" "    ingested               13" \
	"    in a page               6" "    on no source note      82   78 cite source notes that are not in the vault, 4 cite nothing" \
	"Zettel notes no page includes (104)"
run_status 0 $VAULT note "$core" "2412.14093 - Alignment Faking in Large Language Models" -f core
expect "$out" "  stage      compiled" "      50_MOC/topics/emergent-misalignment-from-reward-hacking.md  (via alignment-faking-empirical)"
run_status 1 $VAULT new "$core" -f core --type zettel --title "Graders Fail Inaction" --set "Tags=[[x]]"
expect "$out" "too-few-links        links to 0 source notes"
run_status 0 $VAULT new "$core" -f core --type zettel --title "Graders Fail Inaction" --set "Tags=[[x]]" --body "From [[RLHF]]."
expect "$(cat "$core/20_ZETTLEKASTEN/graders-fail-inaction.md")" "Created: " "# Graders Fail Inaction" "From [[RLHF]]."

step "Pages (vault page new, refresh, check)"
pv="$scratch/page-vault"
cp -Rp tests/fixtures/vault "$pv"
ZS="[[cot-monitor-beats-action-monitor]] [[cot-monitorability-vs-faithfulness]] [[obfuscated-reward-hacking]] [[monitorability-tax]] [[reward-hacking-as-specification-gaming]]"
before=$(snapshot "$pv")
run_status 0 $VAULT page new "$pv" "Monitoring Reasoning" -f core --from "$ZS"
expect "$out" "# Monitoring Reasoning" "<!-- vault:zettels -->" "### ai-safety" \
	"- [[2503.11926 - Monitoring Reasoning Models (CoT Obfuscation)]] (cot-monitor-beats-action-monitor" \
	"Not written: add --write to write it."
unchanged "$pv" "$before" "vault page new without --write"
run_status 0 $VAULT page new "$pv" "Monitoring Reasoning" -f core --from "$ZS" --write
expect "$out" "+ 50_MOC/monitoring-reasoning.md" "+ 50_MOC/index.md" "+ 50_MOC/log.md" "check    no new violations"
expect "$(cat "$pv/50_MOC/index.md")" "- [[monitoring-reasoning]]: Monitoring Reasoning (5 zettels, 1 source)"
expect "$(cat "$pv/50_MOC/log.md")" "page new [[monitoring-reasoning]] (5 zettels, 1 source)"
run_status 1 $VAULT check "$pv" -f core
expect "$out" "✗ 144 violations in 123 notes"
run_status 1 $VAULT page check "$pv" monitoring-reasoning -f core
expect "$out" "✗ behind: 2 zettels with at least two of its tags or links in common, not on it" \
	"20_ZETTLEKASTEN/reward-hacking-as-eval-false-positive.md   tags ai-safety · links obfuscated-reward-hacking"
reject "$out" "block out of date" "changed since the last refresh"
printf '\nMy own text, kept byte for byte.\n' >>"$pv/50_MOC/monitoring-reasoning.md"
sed -i 's/^The monitorability tax is the cost you pay/The monitorability tax is the capability you give up/' "$pv/20_ZETTLEKASTEN/monitorability-tax.md"
run_status 1 $VAULT page check "$pv" monitoring-reasoning -f core
expect "$out" "✗ zettels block out of date" "changed       20_ZETTLEKASTEN/monitorability-tax.md"
run_status 0 $VAULT page refresh "$pv" monitoring-reasoning -f core --write
expect "$out" "rebuilt the zettels block (5 zettels)" "state    .vaultengine/pages.json"
expect "$(tail -2 "$pv/50_MOC/monitoring-reasoning.md")" "My own text, kept byte for byte."
run_status 1 $VAULT page check "$pv" monitoring-reasoning -f core
reject "$out" "block out of date" "changed since the last refresh"
before=$(snapshot "$pv")
run_status 0 $VAULT page exclude "$pv" monitoring-reasoning model-awareness-of-intent-violation-while-cheating -f core
unchanged "$pv" "$before" "vault page exclude without --write"
run_status 0 $VAULT page exclude "$pv" monitoring-reasoning model-awareness-of-intent-violation-while-cheating -f core --write
expect "$out" "Recorded in the engine state (page exclude)" "1 excluded"
mkdir -p "$pv/50_MOC/topics" && mv "$pv/50_MOC/monitoring-reasoning.md" "$pv/50_MOC/topics/cot-monitoring.md"
run_status 1 $VAULT page check "$pv" cot-monitoring -f core
expect "$out" "= renamed from 50_MOC/monitoring-reasoning.md: its record followed it" "= 1 zettel excluded from it"
reject "$out" "no record of its zettels" "model-awareness-of-intent-violation-while-cheating"

step "Pages and the pipeline from the index (vault index; reads parse only what changed)"
$VAULT index "$pv" >/dev/null
run_status 1 $VAULT page check "$pv" cot-monitoring -f core
reject "$out" "no usable index"
expect "$out" "= renamed from 50_MOC/monitoring-reasoning.md"
echo "An edit after the index." >>"$pv/20_ZETTLEKASTEN/reward-hacking-as-eval-false-positive.md"
with_index=$($VAULT pipeline "$pv" -f core --json 2>&1)
mv "$pv/.vaultengine/index.sqlite" "$scratch/index.sqlite.kept"
without_index=$($VAULT pipeline "$pv" -f core --json 2>/dev/null)
mv "$scratch/index.sqlite.kept" "$pv/.vaultengine/index.sqlite"
[[ $with_index == "$without_index" ]] || fail "vault pipeline from the index differs from reading every file"
ok "vault pipeline from a stale index equals reading every file"

step "Findings and ideas (vault findings, vault idea new, vault idea set)"
fv="$scratch/findings-vault"
cp -Rp tests/fixtures/vault "$fv"
$VAULT index "$fv" >/dev/null
before=$(snapshot "$fv")
run_status 0 $VAULT findings "$fv" -f core --limit 1
expect "$out" "missing-note: a note many notes link to that does not exist (72)" \
	"missing-32eacad6a1  [[reward-model-overoptimization-hump]] does not exist; 12 notes link to it" \
	"name-5553fbc478  [[on-policy-vs-off-policy-learning]] is probably [[on-policy-versus-off-policy-learning]]" \
	"link-6b21be7696  [[in-context-scheming-capability]] and [[scheming-inability-safety-case-closed]]"
run_status 0 $VAULT idea new "$fv" --from link-6b21be7696 -f core
expect "$out" "- id: link-6b21be7696" "Status: #open" "Not written: add --write to write it."
unchanged "$fv" "$before" "vault findings and vault idea new without --write"
run_status 0 $VAULT idea new "$fv" --from link-6b21be7696 -f core --write
expect "$out" "+ 50_MOC/ideas/in-context-scheming-capability and scheming-inability-safety-case-closed.md" "check    no new violations"
run_status 1 $VAULT check "$fv" -f core
expect "$out" "✗ 144 violations in 123 notes"
run_status 1 $VAULT new "$fv" -f core --type idea --title "Just a thought" --set "Tags=[[x]]"
expect "$out" "missing-block        no finding block"
run_status 0 $VAULT idea set "$fv" "in-context-scheming-capability and scheming-inability-safety-case-closed" done -f core --write
run_status 1 $VAULT idea set "$fv" "in-context-scheming-capability and scheming-inability-safety-case-closed" parked -f core --write
expect "$out" "Status: #parked is not an allowed value (open, dropped, done)"
run_status 0 $VAULT findings "$fv" -f core --id link-6b21be7696
expect "$out" "link-6b21be7696"

step "Move, merge and the queue (vault merge, vault move, vault next)"
mv_vault="$scratch/move-vault"
cp -Rp tests/fixtures/vault "$mv_vault"
$VAULT index "$mv_vault" >/dev/null
before=$(snapshot "$mv_vault")
run_status 0 $VAULT merge "$mv_vault" on-policy-versus-off-policy-learning on-policy-vs-off-policy-learning -f core
expect "$out" "Would merge: links to [[on-policy-vs-off-policy-learning]] (no note) -> 20_ZETTLEKASTEN/on-policy-versus-off-policy-learning.md" \
	"links    5 links rewritten in 5 notes" "check    no new violations" "Not written: add --write to write it."
unchanged "$mv_vault" "$before" "vault merge without --write"
run_status 0 $VAULT merge "$mv_vault" on-policy-versus-off-policy-learning on-policy-vs-off-policy-learning -f core --write
expect "$out" "Did merge:" "index    updated 5 notes"
expect "$($VAULT health "$mv_vault")" "unresolved links                      1310   across 611 missing targets" \
	"ambiguous links                          0" "missing heading/block targets            0"
run_status 0 $VAULT move "$mv_vault" kl-penalty-in-rlhf kl-penalty-in-rlhf-renamed -f core --write
expect "$out" "rename 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md -> 20_ZETTLEKASTEN/kl-penalty-in-rlhf-renamed.md" "check    no new violations"
expect "$($VAULT health "$mv_vault")" "unresolved links                      1310" "ambiguous links                          0"
run_status 1 $VAULT move "$mv_vault" kl-penalty-in-rlhf-renamed 20_ZETTLEKASTEN/drafts/ -f core --write
expect "$out" "untyped-note"
run_status 0 $VAULT next "$mv_vault" -f core
expect "$out" " 1. write    write [[transcript-spotting-heuristics-for-reward-hacking]]" "run  vault new" "By kind (listed / all):"
run_status 0 $VAULT note "$mv_vault" kl-penalty-in-rlhf-renamed -f core
expect "$out" "trust      "
# A merge drops the links that would point at the note they are in (here a list item holding only [[B]] in A).
run_status 0 $VAULT merge "$mv_vault" enterprise-agent-evaluation-challenges pass-hat-k-reliability-metric -f core
expect "$out" "self     1 link would point at the note it is in: dropped" \
	"20_ZETTLEKASTEN/enterprise-agent-evaluation-challenges.md:31  [[pass-hat-k-reliability-metric]] -> line removed"
run_status 0 $VAULT merge "$mv_vault" enterprise-agent-evaluation-challenges pass-hat-k-reliability-metric -f core --write
survivor=$(cat "$mv_vault/20_ZETTLEKASTEN/enterprise-agent-evaluation-challenges.md")
expect "$survivor" "## Merged from pass-hat-k-reliability-metric"
reject "$survivor" "[[pass-hat-k-reliability-metric]]" "[[enterprise-agent-evaluation-challenges]]"
expect "$($VAULT health "$mv_vault")" "ambiguous links                          0" "missing heading/block targets            0"

step "Findings scale, and are the pair-by-pair implementation's (tests/fixtures/findings/sample_reference.json)"
$PY -m pytest -q tests/test_vaultengine_findings.py -k SameAsPairByPair

step "The reading workflow of docs/workflow.md (ingest, new, note, pipeline, page, findings, idea, next): its commands"
wf="$scratch/wf-vault"
cp -Rp tests/fixtures/vault "$wf"
run_status 0 $VAULT index "$wf"
run_status 0 $VAULT check "$wf" -f core --update-baseline
expect "$out" "Recorded a baseline of 144 violations"
printf '%s\n' '---' 'title: Verifier Gaming in Agentic RL' 'url: https://example.org/verifier-gaming' '---' \
	'Agents trained against automatic verifiers learn to satisfy the verifier rather than the task. The paper' \
	'collects 40 cases across coding and web environments and sorts them into three kinds: tests the agent can' \
	'edit, graders that read only the final message, and rubrics scored by a model that can be flattered.' '' \
	'A do-nothing baseline catches most of them: if an agent that takes no action scores above zero, the' \
	'grader is checking the wrong thing.' >"$scratch/paper.md"
run_status 1 $VAULT ingest "$wf" "$scratch/paper.md" -f core --as source --dry-run
expect "$out" "2  missing-value        'Tags' is empty"
T="[[reward-hacking]] [[evaluation-methodology]]"
run_status 0 $VAULT ingest "$wf" "$scratch/paper.md" -f core --as source --set "Tags=$T" --dry-run
expect "$out" "+Source: https://example.org/verifier-gaming" "Dry run: nothing was written."
run_status 0 $VAULT ingest "$wf" "$scratch/paper.md" -f core --as source --set "Tags=$T"
expect "$out" "+ 10_SOURCES/Verifier Gaming in Agentic RL.md" "check    no new violations" "index    updated 1 note"
run_status 1 $VAULT new "$wf" -f core --type zettel --title "Graders should see the final state" --set "Tags=[[reward-hacking]]" \
	--body "Check what the agent left, not what it said." --dry-run
expect "$out" "too-few-links        links to 0 source notes; a zettel needs at least 1"
run_status 0 $VAULT new "$wf" -f core --type zettel --title "A do-nothing agent is the cheapest grader test" --set "Tags=$T" \
	--body "If an agent that takes no action scores above zero, the grader checks the wrong thing. From [[Verifier Gaming in Agentic RL]]."
expect "$out" "+ 20_ZETTLEKASTEN/a-do-nothing-agent-is-the-cheapest-grader-test.md"
run_status 0 $VAULT new "$wf" -f core --type zettel --title "Editable tests are a reward channel" --set "Tags=$T" \
	--body "When the agent can write to the test files, passing the tests stops meaning the task is done. From [[Verifier Gaming in Agentic RL]]."
run_status 0 $VAULT new "$wf" -f core --type zettel --title "Final-message graders reward confident summaries" --set "Tags=$T" \
	--body "A grader that reads only the last message pays for claims, not for the state the agent left. From [[Verifier Gaming in Agentic RL]]."
run_status 0 $VAULT note "$wf" "Verifier Gaming in Agentic RL" -f core
expect "$out" "stage      extracted" "zettels    3 zettels"
run_status 0 $VAULT pipeline "$wf" -f core --limit 3
expect "$out" "    extracted              13   a zettel links to it, no page uses those yet" "Zettel notes no page includes (107)"
Z="[[a-do-nothing-agent-is-the-cheapest-grader-test]] [[editable-tests-are-a-reward-channel]] [[final-message-graders-reward-confident-summaries]]"
before=$(snapshot "$wf")
run_status 0 $VAULT page new "$wf" "Grader gaming" --from "$Z" -f core
expect "$out" "Would write 3 notes (page new):" "Not written: add --write to write it."
unchanged "$wf" "$before" "vault page new without --write"
run_status 0 $VAULT page new "$wf" "Grader gaming" --from "$Z" -f core --write
expect "$out" "+ 50_MOC/grader-gaming.md" "check    no new violations"
run_status 1 $VAULT page check "$wf" grader-gaming -f core
expect "$out" "20_ZETTLEKASTEN/reward-hacking-as-eval-false-positive.md   tags reward-hacking, evaluation-methodology"
# Your paragraph under the title links the zettel the page is behind on (docs/workflow.md, step 4).
$PY - "$wf/50_MOC/grader-gaming.md" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
para = ("Every grader here checks a proxy the agent can reach, so a pass is a false positive of the eval:\n"
        "[[reward-hacking-as-eval-false-positive]].\n")
open(path, "w", encoding="utf-8").write(text.replace("# Grader gaming\n", "# Grader gaming\n\n" + para, 1))
PYEOF
run_status 1 $VAULT page check "$wf" grader-gaming -f core
expect "$out" "✗ zettels block out of date" "✗ behind: 7 zettels"
run_status 0 $VAULT page refresh "$wf" grader-gaming -f core --write
expect "$out" "rebuilt the zettels and sources blocks (4 zettels)"
run_status 0 $VAULT page exclude "$wf" grader-gaming environment-and-eval-are-the-same-object eval-precision-vs-eval-validity \
	grader-failure-routing llm-judge-position-bias measurement-correction-inflates-apparent-agent-progress \
	model-awareness-of-intent-violation-while-cheating ostensible-positives-from-low-entropy-answer-spaces -f core --write
expect "$out" "50_MOC/grader-gaming.md: 4 zettels recorded, 7 excluded"
run_status 0 $VAULT page check "$wf" grader-gaming -f core
expect "$out" "✓ ready to write: blocks up to date, zettels as recorded, nothing behind, in the index, no violations"
run_status 0 $VAULT findings "$wf" -f core --method link --limit 2
expect "$out" "link-e0e1ed4ecd  [[a-do-nothing-agent-is-the-cheapest-grader-test]] and [[final-message-graders-reward-confident-summaries]]" \
	"cited together by: grader-gaming"
run_status 0 $VAULT idea new "$wf" --from link-e0e1ed4ecd --title "Graders that never look at the state" -f core
expect "$out" "- id: link-e0e1ed4ecd" "Not written: add --write to write it."
run_status 0 $VAULT idea new "$wf" --from link-e0e1ed4ecd --title "Graders that never look at the state" -f core --write
expect "$out" "+ 50_MOC/ideas/Graders that never look at the state.md"
run_status 0 $VAULT next "$wf" -f core --limit 3
expect "$out" "1. write    write [[transcript-spotting-heuristics-for-reward-hacking]]" \
	"idea        1 / 1    open ideas nothing builds on"
run_status 0 $VAULT check "$wf" -f core --baseline
expect "$out" "✓ no new violations in 188 notes checked (144 known violations in the baseline not shown)"

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

step "JSON-RPC reading-workflow methods (vault serve --format core)"
core_rpc="$scratch/core-rpc-vault"
cp -Rp tests/fixtures/vault "$core_rpc"
$VAULT index "$core_rpc" >/dev/null
before=$(snapshot "$core_rpc")
rpc=$(printf '%s\n' \
	'{"jsonrpc":"2.0","id":1,"method":"vault.check","params":{"baseline":true}}' \
	'{"jsonrpc":"2.0","id":2,"method":"vault.pipeline","params":{"limit":1}}' \
	'{"jsonrpc":"2.0","id":3,"method":"findings.list","params":{"methods":["link"],"limit":1}}' \
	'{"jsonrpc":"2.0","id":4,"method":"note.merge","params":{"survivor":"enterprise-agent-evaluation-challenges","other":"pass-hat-k-reliability-metric","dry_run":true}}' \
	'{"jsonrpc":"2.0","id":5,"method":"page.check","params":{}}' \
	'{"jsonrpc":"2.0","id":6,"method":"vault.next","params":{"limit":1}}' |
	$VAULT serve "$core_rpc" --format core 2>/dev/null)
expect "$rpc" '"id":1,"error":{"code":-32008' '"stages":{"ingested":13,"extracted":12,"compiled":3}' \
	'"total":13,"by_method":{"link":13}' '"action":"line removed"' '"managed":0' '"id":6,"result":{"format":{"name":"core"'
unchanged "$core_rpc" "$before" "reads and dry runs over JSON-RPC"
rpc=$(printf '%s\n' \
	'{"jsonrpc":"2.0","id":1,"method":"vault.check","params":{"update_baseline":true,"limit":1}}' \
	'{"jsonrpc":"2.0","id":2,"method":"vault.check","params":{"baseline":true}}' |
	$VAULT serve "$core_rpc" --format core 2>/dev/null)
expect "$rpc" '"recorded_now":true' '"id":2,"result":{"format":{"name":"core"' '"ok":true'

if [[ $QUICK == 0 ]]; then
	step "App: generated types, unit, frame and integration tests (npm test)"
	(cd tui && npm test)
fi

step "App: type check"
(cd tui && npm run typecheck)

step "App: end to end in a real pseudo-terminal (/health; /check, /coverage, /link, /check changed; /explore, keep, link it, /kept, /map since; /pipeline, /findings, /merge, /check baseline, /next)"
(cd tui && npm run test:e2e)

printf '\n\033[32mALL CHECKS PASSED\033[0m\n'
