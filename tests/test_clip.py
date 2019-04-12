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
    expected = np.minimum(amax, np.maximum(arr, amin))
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
    expected = np.minimum(amax, np.maximum(arr, amin))
    assert_equal(result, expected)

# next, mix floats and ints & scalars & arrays
@given(hynp.arrays(np.int64, 10),
       st.floats(),
       hynp.arrays(np.int32, 10),
       )
def test_clip_mixed_arr_scalar(arr, amin, amax):
    result = np.clip(arr, amin, amax)
    # preserve shape on clip
    assert result.shape == arr.shape
    expected = np.minimum(amax, np.maximum(arr, amin))
    assert_equal(result, expected)

# try datetime64 clipping
@given(hynp.arrays(np.int64, 10),
       st.integers(min_value=np.iinfo(np.int64).min,
                   max_value=np.iinfo(np.int64).max),
       st.integers(min_value=np.iinfo(np.int64).min,
                   max_value=np.iinfo(np.int64).max),
       )
def test_clip_timedelta64(arr, amin, amax):
    arr = np.array(arr.tolist(), dtype='m8')
    result = np.clip(arr, amin, amax)
    # preserve shape on clip
    assert result.shape == arr.shape
    # the usual equivalence condition for clip()
    expected = np.minimum(amax, np.maximum(arr, amin))
    assert_equal(result, expected)
