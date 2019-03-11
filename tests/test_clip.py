import numpy as np
from numpy.testing import assert_equal
import hypothesis
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hynp

@given(hynp.arrays(np.int64, 10),
       st.integers(),
       st.integers(),
       )
def test_clip_scalar_ints(arr, amin, amax):
    result = np.clip(arr, amin, amax)
    # preserve shape on clip
    assert result.shape == arr.shape
    # we don't actually check if amin < amax;
    # instead, we test for the property described
    # in numpy/core/code_generators/ufunc_docstrings.py
    # see also: related discussion in NumPy gh-12519
    expected = np.maximum(amin, np.minimum(arr, amax))
    # we also have to convert to lists for the comparison
    # here because we can have object type in output
    # which causes issues
    assert result.tolist() == expected.tolist()

# next, move to the case of floats & arrays
# for the clip limits
@given(hynp.arrays(np.float64, 10),
       hynp.arrays(np.float64, 10),
       hynp.arrays(np.float64, 10),
       )
def test_clip_array_floats(arr, amin, amax):
    result = np.clip(arr, amin, amax)
    # preserve shape on clip
    assert result.shape == arr.shape
    expected = np.maximum(amin, np.minimum(arr, amax))
    assert_equal(result, expected)
