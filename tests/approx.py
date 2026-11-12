# ABOUTME: Compare recorded JSON-like values with a small tolerance on floats, for recordings and golden files.
# ABOUTME: Embeddings are float32 and their last digits can differ between platforms (BLAS, CPU); nothing else may.
"""``same(actual, expected)``: equal, except floats that agree within ``REL``/``ABS``.

Dicts must have the same keys and lists the same length; ints, strings, bools and None compare exactly
(``True`` is not ``1.0``). Used by the recorded JSON-RPC sessions and the explore golden file.
"""

import math

REL = 1e-6
ABS = 1e-7


def same(actual, expected, rel: float = REL) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        if isinstance(actual, bool) or isinstance(expected, bool):
            return actual == expected
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            return False
        return math.isclose(actual, expected, rel_tol=rel, abs_tol=ABS)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same(actual[k], expected[k], rel) for k in actual)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(same(a, e, rel) for a, e in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def assert_same(test, actual, expected, rel: float = REL) -> None:
    """``test.assertEqual`` (with its diff) unless the two are the same within the float tolerance."""
    if not same(actual, expected, rel):
        test.assertEqual(actual, expected)
