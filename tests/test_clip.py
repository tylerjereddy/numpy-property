import numpy as np
import hypothesis
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hynp

@given(hynp.arrays(np.int64, 10),
       st.integers(),
       st.integers(),
       )
def test_clip(arr, amin, amax):
    result = np.clip(arr, amin, amax)
    # preserve shape on clip
    assert result.shape == arr.shape
    # respect clip limits
    assert result.max() <= amax
    assert result.min() >= amin
