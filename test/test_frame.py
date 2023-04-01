import pytest

from src.datapoint import frame_of, minimums, maximums, Point


@pytest.fixture
def points() -> tuple[Point, Point]:
    return Point([-3, 5, 17]), Point([5, 5, -10])


def test_minimums(points: tuple[Point, Point]):
    assert minimums(points) == (-3, 5, -10)


def test_maximums(points: tuple[Point, Point]):
    assert maximums(points) == (5, 5, 17)


def test_frame(points: tuple[Point, Point]):
    frame_points = frame_of(points)

    assert frame_points[0].coords == (-3, 5, -10)
    assert frame_points[1].coords == (5, 5, 17)


def test_frame_types(points: tuple[Point, Point]):
    frame_points = frame_of(points)

    assert type(frame_points[0]) == Point
    assert type(frame_points[1]) == Point



