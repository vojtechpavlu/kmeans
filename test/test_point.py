import pytest

from src.datapoint import Point


@pytest.fixture
def single_point() -> Point:
    return Point([0, 0])


@pytest.fixture
def default_frame() -> tuple[Point, Point]:
    return Point([1, 2]), Point([3, 6])


def test_single_point_creation(single_point: Point):
    assert single_point
    assert single_point.coords == (0, 0)


def test_single_point_dimensionality(single_point: Point):
    assert single_point.dimensionality == 2


def test_single_point_normalization(default_frame: tuple[Point, Point]):
    zeros_point = Point([1, 2])
    ones_point = Point([3, 6])

    between_point1 = Point([2, 4])
    between_point2 = Point([2, 6])
    between_point3 = Point([3, 4])
    between_point4 = Point([1, 4])

    assert zeros_point.normalize(default_frame).coords == (0, 0)
    assert ones_point.normalize(default_frame).coords == (1, 1)

    assert between_point1.normalize(default_frame).coords == (0.5, 0.5)
    assert between_point2.normalize(default_frame).coords == (0.5, 1)
    assert between_point3.normalize(default_frame).coords == (1, 0.5)
    assert between_point4.normalize(default_frame).coords == (0, 0.5)


def test_single_point_repr(single_point: Point):
    assert str(single_point) == "(0, 0)"


