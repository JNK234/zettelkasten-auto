# Set up the sandbox: Python env (uv sync), the terminal app's node_modules (npm ci), and PATH.
#
#   source scripts/setup.sh
#
# Meant to be sourced (it activates .venv in your shell); running it with bash works too.
# No `set -e`: a failure returns from this script instead of closing the shell that sourced it.
# Paths are resolved from this script's location (the repository root is its parent folder), so it works from
# any folder.

_zk_src="${BASH_SOURCE[0]:-$0}"
ZK_ROOT="$(cd "$(dirname "$_zk_src")/.." && pwd)"
ZK_UV="$(command -v uv)"

_zk_setup() {
	echo "== uv sync  ($ZK_ROOT)"
	(cd "$ZK_ROOT" && "$ZK_UV" sync) || { echo "setup: uv sync failed" >&2; return 1; }

	echo "== npm ci  ($ZK_ROOT/tui)"
	# node-pty is built with node-gyp, which downloads the Node.js headers from nodejs.org. That host does not
	# resolve in the sandbox, so use the headers installed with Node when they are there (docs/install.md).
	if [ -f /usr/include/node/node.h ]; then
		(cd "$ZK_ROOT/tui" && npm_config_nodedir=/usr npm ci) || { echo "setup: npm ci failed" >&2; return 1; }
	else
		(cd "$ZK_ROOT/tui" && npm ci) || { echo "setup: npm ci failed" >&2; return 1; }
	fi

	# shellcheck disable=SC1091
	source "$ZK_ROOT/.venv/bin/activate"
	echo "== ready: $(command -v vault), python $("$ZK_ROOT/.venv/bin/python" -c 'import sys; print(sys.version.split()[0])'), node $(node --version)"
}

_zk_setup
