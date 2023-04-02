import pytest

from src.datapoint import Example


@pytest.fixture
def example() -> Example:
    return Example([0, 0])


@pytest.fixture
def example_w_attrs() -> Example:
    return Example([0, 0], {"a": "1", "b": "2"})


@pytest.fixture
def default_frame() -> tuple[Example, Example]:
    return Example([1, 2]), Example([3, 6])


def test_single_example_creation(example_w_attrs: Example):
    assert example_w_attrs
    assert example_w_attrs.coords == (0, 0)
    assert example_w_attrs.other_attrs["a"] == "1"
    assert example_w_attrs.other_attrs["b"] == "2"


def test_single_example_attrs_immutability(example_w_attrs: Example):
    example_w_attrs.other_attrs["a"] = "new value"
    assert example_w_attrs.other_attrs["a"] == "1"


def test_single_example_dimensionality(example: Example):
    assert example.dimensionality == 2


def test_single_example_normalization(default_frame: tuple[Example, Example]):
    zeros_point = Example([1, 2])
    ones_point = Example([3, 6])

    between_point1 = Example([2, 4])
    between_point2 = Example([2, 6])
    between_point3 = Example([3, 4])
    between_point4 = Example([1, 4])

    assert zeros_point.normalize(default_frame).coords == (0, 0)
    assert ones_point.normalize(default_frame).coords == (1, 1)

    assert between_point1.normalize(default_frame).coords == (0.5, 0.5)
    assert between_point2.normalize(default_frame).coords == (0.5, 1)
    assert between_point3.normalize(default_frame).coords == (1, 0.5)
    assert between_point4.normalize(default_frame).coords == (0, 0.5)


def test_example_normalization_type(default_frame: tuple[Example, Example]):
    single_example = Example([1, 1])
    assert type(single_example.normalize(default_frame)) == Example


def test_single_example_repr(example_w_attrs: Example):
    assert str(example_w_attrs) == "(0, 0) - {'a': '1', 'b': '2'}"


