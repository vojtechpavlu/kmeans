import pytest

from src.datapoint import (
    Centroid, Point, NormalizationError, InconsistentDimensionalityError)
from src.metric import Euclidean


@pytest.fixture
def points() -> tuple[Point, Point]:
    return Point([0, 0]), Point([1, 1])


@pytest.fixture
def divergent_points() -> tuple[Point, Point]:
    return Point([4, 0]), Point([-2, 3])


def test_success_creation_using_initor(points: tuple[Point, Point]):
    c = Centroid(coords=[0.5, 0.5], points=points, name="test-centroid")

    assert c
    assert c.coords == (0.5, 0.5)
    assert c.number_of_points == 2
    assert c.dimensionality == 2
    assert c.name == "test-centroid"


def test_success_creation_using_factory(points: tuple[Point, Point]):
    c = Centroid.centroid_of(points=points, name="test-centroid")

    assert c
    assert c.coords == (0.5, 0.5)
    assert c.number_of_points == 2
    assert c.dimensionality == 2
    assert c.name == "test-centroid"


def test_raise_error_on_factory_with_inconsistent_dimensions():
    test_points = Point([0]), Point([1, 1])
    with pytest.raises(InconsistentDimensionalityError) as e:
        Centroid.centroid_of(test_points, name="failure")


def test_automatic_unique_name_assignment(points: tuple[Point, Point]):
    c1 = Centroid.centroid_of(points)
    c2 = Centroid.centroid_of(points)

    assert c1.name != c2.name


def test_repr(points: tuple[Point, Point]):
    c = Centroid.centroid_of(points=points, name="test-centroid")
    assert str(c) == "test-centroid (0.5, 0.5): (points = 2)"


def test_raise_error_on_normalization(points: tuple[Point, Point]):
    with pytest.raises(
            expected_exception=NormalizationError,
            match="Centroids cannot be normalized"):
        Centroid.centroid_of(points).normalize(points)


def test_frame(divergent_points: tuple[Point, Point]):
    mins, maxs = Centroid.centroid_of(divergent_points).frame
    assert mins.coords == (-2, 0)
    assert maxs.coords == (4, 3)


def test_variance(divergent_points: tuple[Point, Point]):
    c = Centroid.centroid_of(divergent_points)
    assert round(c.variance(Euclidean()), 2) == 22.5

