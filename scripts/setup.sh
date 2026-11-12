# Install everything and put `vault` (and `vault-tui`) on PATH, from this checkout.
#
#   source scripts/setup.sh
#
# Meant to be sourced: no `set -e` (it would close your shell on the first error), your current folder is
# left alone, and a failed step stops the script with `return 1`. Safe to source again; every step is
# idempotent (`uv sync` is a no-op when nothing changed, `npm ci` reinstalls from the lock file).
# Run as a plain script (`bash scripts/setup.sh`) it installs the same, but cannot activate the venv for you.
# Steps are the ones in docs/install.md.

# The repository root is this script's parent folder, so it works from any folder.
_zk_src="${BASH_SOURCE[0]:-$0}"
ZK_REPO="$(cd "$(dirname "$_zk_src")/.." && pwd)"

_zk_setup() {
	# 1. Python 3.12 env in $ZK_REPO/.venv: engine, `vault` CLI, test deps, and the zettel tool (torch etc.;
	#    several GB, a few minutes on the first run).
	echo "== uv sync ($ZK_REPO/.venv)"
	uv sync --project "$ZK_REPO" || { echo "setup: uv sync failed" >&2; return 1; }

	# 2. The terminal app. node-pty is compiled by node-gyp, which downloads Node headers from nodejs.org;
	#    offline (as in this sandbox) point it at the headers installed with Node instead.
	echo "== npm ci ($ZK_REPO/tui)"
	if [ -f /usr/include/node/node.h ]; then
		npm_config_nodedir=/usr npm ci --prefix "$ZK_REPO"/tui ||
			{ echo "setup: npm ci failed" >&2; return 1; }
	else
		npm ci --prefix "$ZK_REPO"/tui || { echo "setup: npm ci failed" >&2; return 1; }
	fi
	echo "== npm run build && npm link (vault-tui on PATH)"
	npm run build --prefix "$ZK_REPO"/tui || { echo "setup: tui build failed" >&2; return 1; }
	(cd "$ZK_REPO"/tui && npm link) || { echo "setup: npm link failed" >&2; return 1; }

	# 3. `vault` on PATH for this shell.
	# shellcheck disable=SC1091
	. "$ZK_REPO"/.venv/bin/activate || { echo "setup: cannot activate the venv" >&2; return 1; }
	echo "== ready: $(command -v vault), $(command -v vault-tui)"
	echo "   tests: (cd $ZK_REPO && python -m pytest -q)   # about 8 minutes"
}

_zk_setup
_zk_status=$?
unset -f _zk_setup
# `return` works when sourced; `exit` when run as a script.
return $_zk_status 2>/dev/null || exit $_zk_status
