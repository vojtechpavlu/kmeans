import pytest

from src.datapoint import (
    Point, dimensionality_check, InconsistentDimensionalityError)


@pytest.fixture
def consistent_points() -> tuple[Point, Point]:
    return Point([0, 0]), Point([1, 1])


@pytest.fixture
def inconsistent_points() -> tuple[Point, Point]:
    return Point([0]), Point([1, 1])


def test_doesnt_raise_error_when_check_consistent(
        consistent_points: tuple[Point, Point]):
    dimensionality_check(consistent_points)


def test_does_raise_error_when_check_inconsistent(
        inconsistent_points: tuple[Point, Point]):
    with pytest.raises(
            expected_exception=InconsistentDimensionalityError,
            match="Inconsistent dimensionality of points"):
        dimensionality_check(inconsistent_points)


def test_does_raise_error_when_empty_points():
    with pytest.raises(
            expected_exception=ValueError,
            match="No point provided"):
        dimensionality_check([])



