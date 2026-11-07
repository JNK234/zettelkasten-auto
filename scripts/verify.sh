#!/usr/bin/env bash
# Run every check of vaultengine + vault CLI + vault-tui and stop at the first failure.
# Needs the installs from README.md ("Install"): `uv sync` and `cd tui && npm ci`.
# Works on a scratch copy of the sample vault; nothing in the repo is modified.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
VAULT=.venv/bin/vault
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
sample="$scratch/sample-vault"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
# expect "<output>" "<line that must appear>"...
expect() {
	local output=$1
	shift
	for wanted in "$@"; do
		if ! grep -qF -- "$wanted" <<<"$output"; then
			printf '\033[31mFAIL\033[0m expected to see: %s\n--- got:\n%s\n' "$wanted" "$output"
			exit 1
		fi
		printf '  ok  %s\n' "$wanted"
	done
}

step "Python: engine, index, JSON-RPC and legacy tests"
$PY -m pytest -q

step "CLI on a copy of the sample vault"
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
	'"id":11,"error":{"code":-32003,"message":"this ( is never closed","data":{"query":"reward (tag:ai-safety OR path:50_MOC","start":7,"end":8,"hint":"add ) after the group"}}' \
	'"id":12,"result":{"path":"20_ZETTLEKASTEN/monitorability-tax.md","title":"monitorability-tax","query":"monitorability","matched":true' \
	'"title":"RLHF","score":' \
	'"id":14,"result":{"note":null,"path":"20_ZETTLEKASTEN","links":[{"a":"20_ZETTLEKASTEN/in-context-scheming-capability.md","b":"20_ZETTLEKASTEN/sandbagging-strategic-underperformance.md"' \
	'"id":15,"result":{"total_missing":612,"notes":[{"name":"rl-environment-design","linking_notes":52' \
	'"id":16,"result":{"note":"20_ZETTLEKASTEN/monitorability-tax.md","total":4,"notes":1' \
	'"id":17,"error":{"code":-32002,"message":"no note or link target called \"no-such-note\""'

step "TUI: generated types, unit, frame and integration tests"
(cd tui && npm test)

step "TUI: type check"
(cd tui && npm run typecheck)

step "TUI: end-to-end in a pseudo-terminal (node-pty)"
(cd tui && npm run test:e2e)

printf '\n\033[32mALL CHECKS PASSED\033[0m\n'
