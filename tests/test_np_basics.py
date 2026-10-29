"""
Run with: uv run pytest -q tests/test_np_basics.py
"""
import numpy as np
import pytest

from nano_assistant.np_basics import (
    matmul_shapes,
    one_hot,
    softmax,
    softmax_matrix,
    top_k,
)

RNG = np.random.default_rng(seed=0)


# ---------------------------------------------------------------- softmax
class TestSoftmax:
    def test_sums_to_one(self):
        p = softmax(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(p.sum(), 1.0)

    def test_known_values(self):
        p = softmax(np.array([0.0, 0.0]))
        np.testing.assert_allclose(p, [0.5, 0.5])

    def test_all_positive(self):
        p = softmax(np.array([-5.0, 0.0, 5.0]))
        assert np.all(p > 0)

    def test_order_preserved(self):
        x = np.array([0.1, 3.0, -2.0, 1.0])
        assert np.argmax(softmax(x)) == np.argmax(x)

    def test_numerically_stable(self):
        p = softmax(np.array([1000.0, 1000.0]))
        assert not np.any(np.isnan(p))
        np.testing.assert_allclose(p, [0.5, 0.5])

    def test_shift_invariant(self):
        x = RNG.normal(size=10)
        np.testing.assert_allclose(softmax(x), softmax(x + 42.0))


# --------------------------------------------------------- softmax_matrix
class TestSoftmaxMatrix:
    def test_shape_kept(self):
        X = RNG.normal(size=(4, 7))
        assert softmax_matrix(X).shape == (4, 7)

    def test_each_row_sums_to_one(self):
        X = RNG.normal(size=(4, 7))
        np.testing.assert_allclose(softmax_matrix(X).sum(axis=-1), np.ones(4))

    def test_matches_row_by_row_softmax(self):
        X = RNG.normal(size=(3, 5))
        expected = np.stack([softmax(row) for row in X])
        np.testing.assert_allclose(softmax_matrix(X), expected)

    def test_stable_per_row(self):
        X = np.array([[1000.0, 1000.0], [-1000.0, -1000.0]])
        np.testing.assert_allclose(softmax_matrix(X), [[0.5, 0.5], [0.5, 0.5]])

    def test_works_on_3d_logits(self):
        """Real LLM logits are (batch, tokens, vocab): softmax must use the LAST axis."""
        logits = RNG.normal(size=(2, 3, 10))
        p = softmax_matrix(logits)
        np.testing.assert_allclose(p.sum(axis=-1), np.ones((2, 3)))


# ---------------------------------------------------------- matmul_shapes
@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((2, 3), (3, 4)),
        ((5, 2, 3), (3, 4)),          # batch of 5 matrices
        ((1, 7, 896), (896, 1000)),   # (batch, tokens, dim) @ (dim, vocab) -> logits
    ],
)
def test_matmul_shapes_matches_numpy(shape_a, shape_b):
    A, B = np.ones(shape_a), np.ones(shape_b)
    assert matmul_shapes(A, B) == (A @ B).shape


def test_matmul_shapes_incompatible_raises():
    with pytest.raises(ValueError):
        matmul_shapes(np.ones((2, 3)), np.ones((4, 5)))


# ------------------------------------------------------------------ top_k
class TestTopK:
    def test_basic(self):
        x = np.array([0.1, 0.9, 0.3, 0.7])
        np.testing.assert_array_equal(top_k(x, 2), [1, 3])

    def test_descending_order(self):
        x = RNG.normal(size=20)
        idx = top_k(x, 5)
        assert np.all(np.diff(x[idx]) <= 0)

    def test_k1_is_argmax(self):
        x = RNG.normal(size=20)
        assert top_k(x, 1)[0] == np.argmax(x)

    def test_k_equals_length_returns_all(self):
        x = RNG.normal(size=6)
        assert sorted(top_k(x, 6)) == list(range(6))

    def test_k0_returns_empty(self):
        assert len(top_k(np.array([1.0, 2.0, 3.0]), 0)) == 0


# ---------------------------------------------------------------- one_hot
class TestOneHot:
    def test_shape(self):
        assert one_hot(np.array([2, 0, 4]), 5).shape == (3, 5)

    def test_exact_values(self):
        expected = [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
        np.testing.assert_array_equal(one_hot(np.array([2, 0, 4]), 5), expected)

    def test_rows_sum_to_one(self):
        oh = one_hot(np.array([1, 1, 3]), 4)
        np.testing.assert_array_equal(oh.sum(axis=1), [1, 1, 1])

    def test_argmax_roundtrip(self):
        idx = RNG.integers(0, 50, size=30)
        np.testing.assert_array_equal(one_hot(idx, 50).argmax(axis=1), idx)

    def test_accepts_python_list(self):
        assert one_hot([2, 0, 4], 5).shape == (3, 5)

    def test_matmul_is_embedding_lookup(self):
        """one_hot(idx) @ E selects the rows of E: this is how token ids become vectors."""
        E = RNG.normal(size=(5, 3))
        idx = np.array([2, 0, 2])
        np.testing.assert_allclose(one_hot(idx, 5) @ E, E[idx])