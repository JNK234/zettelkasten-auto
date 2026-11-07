# Install

[← Documentation index](../README.md#documentation)

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 22 or newer (for `vault-tui`)
- For the app's end-to-end tests: a C/C++ toolchain for `node-pty` (Linux: `build-essential` and `python3`; macOS uses a prebuilt binary)

## Install

From the repository root:

```bash
uv sync                          # Python env in .venv (engine, CLI, tests, and the zettel tool)
source .venv/bin/activate        # puts `vault` on PATH
cd tui && npm ci && npm run build && npm link && cd ..   # puts `vault-tui` on PATH
```

`uv sync` also installs `torch`/`sentence-transformers` for the zettel tool's embeddings (several GB on Linux, about 5 minutes).
The engine, CLI and app don't need them. Optional: `uv pip install wordllama` (or `uv sync --extra semantic`) for the
experimental `vault suggest --method semantic|fused`.

Offline or behind a proxy: `npm ci` builds `node-pty` with node-gyp, which downloads the Node.js headers from nodejs.org.
If that fails (`gyp ERR! … ENOTFOUND`), point it at the headers installed with Node: `npm_config_nodedir=/usr npm ci`
(the folder that contains `include/node/node.h`).


Next: [Getting started](getting-started.md) walks through the sample vault; [Verify it yourself](verify.md) checks the install.
