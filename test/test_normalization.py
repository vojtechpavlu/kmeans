import pytest

from src.datapoint import (
    normalize, Point, InconsistentDimensionalityError, NormalizationError)


@pytest.fixture
def valid_points() -> tuple[Point, Point, Point]:
    return Point([-3, 4, 20]), Point([5, 6, -20]), Point([1, 8, 0])


@pytest.fixture
def inconsistent_dim_points() -> tuple[Point, Point]:
    return Point([-3]), Point([5, 5, -10])


@pytest.fixture
def zero_difference() -> tuple[Point, Point]:
    return Point([1, 5]), Point([1, 8])


def test_interval_of_values(valid_points: tuple[Point, Point, Point]):
    normalized = normalize(valid_points)

    for p in normalized:
        for value in p.coords:
            assert value >= 0 or value <= 1


def test_values_after_normalization(valid_points: tuple[Point, Point, Point]):
    p1, p2, p3 = normalize(valid_points)

    assert p1.coords == (0.0, 0.0, 1.0)
    assert p2.coords == (1.0, 0.5, 0.0)
    assert p3.coords == (0.5, 1.0, 0.5)


def test_raises_error_on_inconsistent(
        inconsistent_dim_points: tuple[Point, Point]):
    with pytest.raises(InconsistentDimensionalityError):
        normalize(inconsistent_dim_points)


def test_raises_error_on_zero_difference_in_one_dim(
        zero_difference: tuple[Point, Point]):
    with pytest.raises(
            expected_exception=NormalizationError,
            match="Difference between min and max has to "
                  "be for every dimension > 0"):
        normalize(zero_difference)
