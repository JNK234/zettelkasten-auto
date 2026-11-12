"""The embedding model the tests use: models/potion-base-8M, the one config/config.yaml points at.

The model is shipped next to the repository, not in git (see .gitignore). Tests that need its
vectors are marked ``@requires_model`` and are skipped, with this reason, when it is not there.
"""

import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "potion-base-8M"
HAVE_MODEL = (MODEL_DIR / "model.safetensors").is_file() and (MODEL_DIR / "tokenizer.json").is_file()

requires_model = unittest.skipUnless(HAVE_MODEL, f"embedding model not found at {MODEL_DIR}")
