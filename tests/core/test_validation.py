import pytest

from perf_benchmark.core.validation import positive_int


@pytest.mark.parametrize("value", [True, False, -1, 0.0, 1.5, "1", None])
@pytest.mark.parametrize("allow_zero", [False, True])
def test_integer_validation_rejects_nonintegers_and_negative_values(value, allow_zero):
    with pytest.raises(ValueError, match="count must be a .*integer"):
        positive_int(value, "count", allow_zero=allow_zero)


def test_integer_validation_preserves_positive_optional_and_nonnegative_bounds():
    assert positive_int(1, "count") == 1
    assert positive_int(0, "count", allow_zero=True) == 0
    assert positive_int(None, "count", optional=True, allow_zero=True) is None
    with pytest.raises(ValueError, match="positive integer"):
        positive_int(0, "count")
