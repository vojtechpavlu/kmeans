"""This module contains all the means for representing and working with
datapoints in the multidimensional space.
"""

from typing import Iterable, Dict
from copy import deepcopy

from typing import TYPE_CHECKING

# Problems with circular imports
if TYPE_CHECKING:
    from src.metric import Metric


class Point:
    """Instances of this class represents the points in the multidimensional
    space. Each of the point has to be able to tell its coordinates in the
    space and its dimensionality.
    """

    def __init__(self, coords: Iterable[float]):
        """Initor accepting the numeric (float) values as the representation
        of the point in the multidimensional space.
        """
        self.__coords = list(coords)

    @property
    def coords(self) -> tuple[float]:
        """Tuple of the values representing the point in the multidimensional
        space.

        For each of the dimension of the point, it returns a single numeric
        value as a representation of indentation in the given axis.
        """
        return tuple(self.__coords)

    @property
    def dimensionality(self) -> int:
        """Number of dimensions this point is described in."""
        return len(self.coords)

    def normalize(self, frame: tuple["Point", "Point"]) -> "Point":
        """Normalizes the values for each of the dimension by the given frame.
        The result values for each of the dimension are inside interval of
        [0; 1].
        """
        normalized = []

        for dim in range(self.dimensionality):
            d_min = frame[0].coords[dim]
            d_max = frame[1].coords[dim]
            normalized.append((self.coords[dim] - d_min) / (d_max - d_min))

        return Point(normalized)

    def __repr__(self):
        return f"{self.coords}"


class Example(Point):
    """Instances of this class represents the examples of the given problem
    as datapoints in the multidimensional space.

    It can also hold some additional data for better description of the
    example. These other attributes are represented as a dictionary of
    strings mapped by strings.
    """

    def __init__(self,
                 coords: Iterable[float],
                 other_attrs: Dict[str, str] = None):
        """Initor accepting the actual point coordinates and an optional
        dictionary of other parameters mapped as `str -> str`.
        """
        super().__init__(coords)
        if other_attrs is None:
            other_attrs = dict()
        self.__other_attrs = other_attrs

    @property
    def other_attrs(self) -> Dict[str, str]:
        """Deep copy of the dictionary responsible for storing other less
        relevant attributes mapped by `str -> str`.
        """
        return deepcopy(self.__other_attrs)

    def normalize(self, frame: tuple["Point", "Point"]) -> "Point":
        """Normalizes the values for each of the dimension by the given frame.
        The result values for each of the dimension are inside interval of
        [0; 1].
        """
        normalized = []

        for dim in range(self.dimensionality):
            d_min = frame[0].coords[dim]
            d_max = frame[1].coords[dim]
            normalized.append((self.coords[dim] - d_min) / (d_max - d_min))

        return Example(normalized, self.other_attrs)

    def __repr__(self):
        return f"{self.coords} - {self.__other_attrs}"


class Centroid(Point):
    """Instances of this class represents the centroids of the point cluster
    in the multidimensional space.

    It can be created either using a constructor or by static method
    `centroid_of`.
    """

    # Counter to uniquely identify the centroids
    __COUNTER = 0

    def __init__(self,
                 coords: Iterable[float],
                 points: Iterable[Point],
                 name: str = ""):
        """Initor accepting the coordinates of this centroid as an
        `Iterable` of numeric values, the points this centroid consists
        of and an optional name of the cluster. If not provided, it will
        be generated as a constant prefix 'cluster_' and iterated number.
        This name should be unique.
        """
        super().__init__(coords)
        self.__points = list(points)
        self.__name = name

        if not self.name:
            self.__name = f"centroid_{Centroid.__COUNTER}"
            Centroid.__COUNTER += 1

    @property
    def points(self) -> tuple[Point]:
        """Tuple of points in this cluster."""
        return tuple(self.__points)

    @property
    def number_of_points(self) -> int:
        """Number of points in this cluster."""
        return len(self.__points)

    @property
    def name(self) -> str:
        """Name given to the cluster."""
        return self.__name

    @property
    def frame(self) -> tuple[Point, Point]:
        """Calculates the two framing points describing the space the
        datapoints being assigned to this cluster (centroid)."""
        return frame_of(self.points)

    def variance(self, metric: "Metric") -> float:
        """Calculates the variance of the cluster by squaring the distances
        between the current centroid coordinates and each of the point
        assigned to it.

        To achieve this approach, the metric to calculate the distance has to
        be provided."""
        dist_squares = [metric.distance(self, p) ** 2 for p in self.points]
        return sum(dist_squares)

    def __repr__(self):
        """String representation of the cluster."""
        return f"{self.name} {self.coords}: (points = {self.number_of_points})"

    def normalize(self, frame: tuple["Point", "Point"]) -> "Point":
        """Raises the exception, because centroids cannot be normalized since
        these are just static points related to the given points."""
        raise NormalizationError("Centroids cannot be normalized")

    def add_point(self, point: Point):
        """Registers the given point in the centroid."""
        self.__points.append(point)

    def flush(self):
        """Removes all the points assigned to the centroid."""
        self.__points.clear()

    def recalculate(self) -> "Centroid":
        """Recalculates the centroid coordinates by creating the new instance.
        This method works just as described in design pattern of State.
        """
        return Centroid.centroid_of(self.points, self.name)

    @staticmethod
    def centroid_of(points: Iterable[Point], name: str = "") -> "Centroid":
        """Static method implemented as a factory for the centroids by given
        points.

        This method is responsible for creating a new instance of centroid to
        make the process of the creation easier.
        """
        # Check the consistency of dimensions
        dimensionality_check(points)

        # Turn points into a tuple
        points = tuple(points)
        n = len(points)

        # Default dimensionality
        dimensionality = points[0].dimensionality
        averages = []

        for dimension in range(dimensionality):
            averages.append(sum(p.coords[dimension] for p in points) / n)

        return Centroid(averages, points, name)

    @staticmethod
    def centroid_of_point(point: Point, name: str = "") -> "Centroid":
        """Static method implemented as a factory for the centroids by given
        point that is copied - in terms of the point's coordinates.

        The given point is transformed into the centroid.
        """
        return Centroid(point.coords, (), name)


class InconsistentDimensionalityError(Exception):
    """This exceptions represents an error when the datapoints the system
    has to work with does not have the same dimensionalities and thus it
    cannot calculate over these points.
    """

    def __init__(self, points: Iterable[Point]):
        """Initor taking the points which are not consistent in terms of their
        dimensionality.
        """
        super().__init__(f"Inconsistent dimensionality of points")
        self.__points = points

    @property
    def points(self) -> tuple[Point]:
        """Tuple of inconsistent datapoints."""
        return tuple(self.__points)


class NormalizationError(Exception):
    """This exception describes the error occurring when the system finds a
    problem when trying to perform a normalization."""

    def __init__(self, message: str):
        """Initor taking the string-based message and passing it to the
        parent class initor (`Exception`).
        """
        super().__init__(message)


def dimensionality_check(points: Iterable[Point]):
    """Checks that all of the given points have the same dimensionality.
    When there is a point with unique number of dimensions, it raises an
    `InconsistentDimensionalityError`.
    """
    if len(tuple(points)) == 0:
        raise ValueError("No point provided")
    elif len({point.dimensionality for point in points}) > 1:
        raise InconsistentDimensionalityError(points)


def minimums(points: Iterable[Point]) -> tuple[float]:
    """Finds out the minimal values in each of the dimension for each of the
    point in the given iterable of points.
    """
    dimensionality_check(points)
    points = tuple(points)
    n_dim = points[0].dimensionality

    return tuple(
        # For each dimension find the minimum over all the points and
        # cast these to float for type hinting purposes
        min([float(p.coords[dim]) for p in points]) for dim in range(n_dim)
    )


def maximums(points: Iterable[Point]) -> tuple[float]:
    """Finds out the maximum values in each of the dimension for each of the
    point in the given iterable of points.
    """
    dimensionality_check(points)
    points = tuple(points)
    n_dim = points[0].dimensionality

    return tuple(
        # For each dimension find the maximum over all the points and
        # cast these to float for type hinting purposes
        max([float(p.coords[dim]) for p in points]) for dim in range(n_dim)
    )


def frame_of(points: Iterable[Point]) -> tuple[Point, Point]:
    """Creates two framing points as a minimum and maximum values in each of
    the dimension."""
    return Point(minimums(points)), Point(maximums(points))


def normalize(points: Iterable[Point]) -> tuple[Point]:
    """Tries to normalize all of the values inside the given iterable of
    points.

    It fails in these cases:
        - When given an empty iterable
        - When all of the points does not have the same dimensionality
        - When for any of the dimension is the min value equal to the max one
        - When is normalization performed on a centroid
    """
    dimensionality_check(points)

    points = tuple(points)

    if len(points) == 0:
        raise NormalizationError("Cannot normalize an empty iterable")

    n_dims = points[0].dimensionality

    frame = frame_of(points)
    mins = frame[0].coords
    maxs = frame[1].coords

    if not all(maxs[dim] != mins[dim] for dim in range(n_dims)):
        raise NormalizationError(
            "Difference between min and max has to be for every dimension > 0")

    return tuple(point.normalize(frame) for point in points)



