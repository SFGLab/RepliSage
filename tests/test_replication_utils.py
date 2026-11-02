"""Replication matrix helpers and the replication simulator."""
import numpy as np
import pytest
from src.MC.replication_matrix import to_time_major
from src.utils.arrays import expand_columns, rescale_matrix

def test_time_major_equals_expand_columns():
    m = (np.random.default_rng(0).random((30, 7)) > 0.5).astype(int)
    for dur in (8, 14, 20, 23):
        assert np.array_equal(to_time_major(m, dur), expand_columns(m, dur).T)

def test_time_major_layout():
    out = to_time_major(np.ones((5, 4), int), 10)
    assert out.dtype == np.int8 and out.shape == (10, 5) and out.flags["C_CONTIGUOUS"]

def test_time_major_requires_longer_duration():
    with pytest.raises(ValueError): to_time_major(np.ones((5, 4), int), 4)

def test_rescale_matrix_shape():
    assert rescale_matrix(np.eye(10), 20).shape[0] in (10, 20)
