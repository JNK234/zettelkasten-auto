#!/usr/bin/env python3
"""Experiment: do embeddings rank better than words for the vault's suggestions?

    python scripts/semantic_eval.py [vault] [--format brain]

A) a note's source, B) the claim an evidence note supports (both from links the vault
already has, with the links removed from the text), C) paraphrases of sample papers.
Needs `pip install wordllama` for the semantic and fused columns.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.INFO)

from vaultengine import Vault  # noqa: E402
from vaultengine.experimental.semantic import evaluate, evaluate_paraphrases, semantic_available  # noqa: E402
from vaultengine.formats import BUNDLED_FORMATS_DIR, TypedVault, load_format, resolve_format_path  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("vault", nargs="?", default=str(Path(__file__).resolve().parent.parent / "tests/fixtures/vault"))
parser.add_argument("--format", default="brain")
args = parser.parse_args()

methods = ("lexical", "semantic", "fused") if semantic_available() else ("lexical",)
typed = TypedVault(load_format(resolve_format_path(args.format, [Path("formats"), BUNDLED_FORMATS_DIR])), Vault.load(args.vault))
started = time.perf_counter()
rows = evaluate(typed, methods)
rows["C_paper_of_paraphrase"] = evaluate_paraphrases(typed, methods)
labels = {"A_source_of_note": "A  note -> its source", "B_claim_of_evidence": "B  evidence -> its claim",
          "C_paper_of_paraphrase": "C  paraphrase -> its paper"}
print(f"{'task':28} {'queries':>7}  " + "  ".join(f"{m:>16}" for m in methods))
for key, label in labels.items():
    row = rows[key]
    cells = "  ".join(f"{row[m]['top1']:>3}/{row['queries']} · {row[m]['top3']:>2} top3" for m in methods)
    print(f"{label:28} {row['queries']:>7}  {cells}")
print(f"(top-1 · top-3 hits; {time.perf_counter() - started:.1f}s)")
if len(methods) == 1:
    print("semantic and fused need WordLlama: pip install wordllama")
