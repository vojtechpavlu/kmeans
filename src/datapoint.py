""""""

from typing import Iterable, Dict
from copy import deepcopy


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
        self.__points = tuple(points)
        self.__name = name
        if not self.name:
            self.__name = f"centroid_{self.__COUNTER}"
            self.__COUNTER += 1

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

    def __repr__(self):
        """String representation of the cluster."""
        return f"{self.name} {self.coords}: (points = {self.number_of_points})"

    @staticmethod
    def centroid_of(points: Iterable[Point], name: str = ""):
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


def dimensionality_check(points: Iterable[Point]):
    """Checks that all of the given points have the same dimensionality.
    When there is a point with unique number of dimensions, it raises an
    `InconsistentDimensionalityError`.
    """
    if len({point.dimensionality for point in points}) > 1:
        raise InconsistentDimensionalityError(points)


