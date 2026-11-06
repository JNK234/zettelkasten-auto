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
expect "$($VAULT index "$sample")" "179 notes, 0 attachments" "parsed     179  (179 added, 0 changed, 0 deleted)" "links      1607 resolved this run"
expect "$($VAULT index "$sample")" "parsed     0  (index already up to date)"
echo "Edited." >>"$sample/20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
expect "$($VAULT index "$sample")" "parsed     1  (0 added, 1 changed, 0 deleted)" "~ 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
expect "$($VAULT stats "$sample")" "notes             179" "resolved        552   unresolved 1055 (581 distinct targets) · ambiguous 0"
expect "$($VAULT health "$sample")" "unresolved links                      1055   across 581 missing targets" \
	"orphans (no backlinks)                   7" "dead ends (no outgoing note links)      22" "empty notes                              0"

step "JSON-RPC 2.0 over stdio (vault serve)"
rpc=$(printf '%s\n' \
	'{"jsonrpc":"2.0","id":1,"method":"notes.pick","params":{"query":"klpen","limit":1}}' \
	'{"jsonrpc":"2.0","id":2,"method":"notes.delete"}' \
	'{"jsonrpc":"1.0","id":3,"method":"index.status"}' |
	$VAULT serve "$sample" 2>/dev/null)
expect "$rpc" '"path":"20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"' '"id":2,"error":{"code":-32601' '"id":3,"error":{"code":-32600'

step "TUI: generated types, unit, frame and integration tests"
(cd tui && npm test)

step "TUI: type check"
(cd tui && npm run typecheck)

step "TUI: end-to-end in a pseudo-terminal (node-pty)"
(cd tui && npm run test:e2e)

printf '\n\033[32mALL CHECKS PASSED\033[0m\n'
