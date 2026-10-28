import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Turn scores (logits) into probabilities along one axis.

    Works for a vector, a matrix, or LLM logits of shape (batch, tokens, vocab):
    by default it normalizes the LAST axis, which is always the vocabulary.

    :param x: Scores, any shape (a list is accepted too).
    :param axis: Axis to normalize. -1 means "the last one".
    :return: Array of the same shape whose values along `axis` sum to 1.
    """
    x = np.asarray(x, dtype=float)
    # Subtracting the max does not change the result (softmax is shift
    # invariant) but prevents exp() from overflowing to inf.
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def softmax_matrix(X: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Kept for compatibility: softmax() already does this."""
    return softmax(X, axis=-1)


def matmul_shapes(A: np.ndarray, B: np.ndarray) -> tuple:
    """Predict the shape of A @ B without computing it.

    Rule: the last dimension of A must equal the second-to-last of B.
    Leading dimensions are "batch" dimensions and are broadcast.
    Example: (5, 2, 3) @ (3, 4) -> (5, 2, 4).

    :param A: First array, at least 2-D.
    :param B: Second array, at least 2-D.
    :return: Shape of A @ B as a tuple.
    :raises ValueError: If the shapes are not compatible.
    """
    A, B = np.asarray(A), np.asarray(B)
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError(f"Both arrays must be at least 2-D, got {A.shape} and {B.shape}.")
    if A.shape[-1] != B.shape[-2]:
        raise ValueError(
            f"Shapes {A.shape} and {B.shape} are not compatible: "
            f"{A.shape[-1]} (last dim of A) != {B.shape[-2]} (second-to-last dim of B)."
        )
    # Batch dimensions follow broadcasting rules; this raises ValueError if they clash.
    batch = np.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    return batch + (A.shape[-2], B.shape[-1])


def top_k(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values, largest first.

    :param x: 1-D array of scores.
    :param k: Number of indices to return (0 <= k).
    :return: Integer array of length min(k, len(x)).
    """
    x = np.asarray(x)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}.")
    # Sort by -x to get descending order directly; slicing [:k] is safe for k=0,
    # unlike [-k:] where -0 == 0 would return the whole array.
    return np.argsort(-x, kind="stable")[:k]


def one_hot(indices: np.ndarray, n: int) -> np.ndarray:
    """One-hot encode integer indices.

    Row i is all zeros except a 1 at column indices[i].
    one_hot(idx, n) @ E equals E[idx]: this is an embedding lookup.

    :param indices: Integer indices in [0, n), array or list.
    :param n: Number of classes (vocabulary size).
    :return: Float array of shape (len(indices), n).
    """
    indices = np.asarray(indices, dtype=int).ravel()
    if indices.size and (indices.min() < 0 or indices.max() >= n):
        raise ValueError(f"Indices must be in [0, {n}), got min={indices.min()}, max={indices.max()}.")
    out = np.zeros((indices.size, n))
    out[np.arange(indices.size), indices] = 1.0
    return out