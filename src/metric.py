"""This module contains the most simple metrics for the distance evaluation.
Metric is a generalization of distance, while distance is a difference between
two points in the space.

See also: https://en.wikipedia.org/wiki/Metric_space
"""

from abc import ABC, abstractmethod

from src.datapoint import Point, dimensionality_check


class Metric(ABC):
    """This abstract class defines the mutual protocol for all of the metrics.
    All of the metrics (in this case distance) are mathematical concepts
    defined by the following rules:

        There is function `d: S x S -> R`, where `s ∈ S`:
            1) `∀ s1, s2 ∈ S, d(s1, s2) >= 0`
            2) `∀ s1, s2 ∈ S, d(s1, s2) = d(s2, s1)`
            3) `∀ s1, s2 ∈ S, d(s1, s2) = 0  <=>  s1 = s2`
            4) `∀ s1, s2, s3 ∈ S, d(s1, s2) + d(s2, s3) >= d(s1, s3)`

    It simply speaking means:

        1) Every distance has to be greater or equal to zero
        2) Distance from one point to another has to be equal, no matter what
           direction are calculating from
        3) When the distance is equal to zero, it means the two points are in
           the given space equal
        4) Triangle inequality - distances between A -> B and B -> C has to be
           always greater or equal (when the vectors are linearly dependent)
           to the distance between A -> C.
    """

    @abstractmethod
    def distance(self, p1: Point, p2: Point) -> float:
        """Calculates the distance between two points. The result is always a
        real number."""


class Euclidean(Metric):
    """Instances of this class are metrics responsible for calculating the
    euclidean distance between two points.

    The euclidean distance is a shortest distance in between two points in
    cartesian space and can be represented as a length of a vector going from
    point A to point B. This distance is calculated using pythagorean theorem
    applied to a multidimensional space.
    """

    def distance(self, p1: Point, p2: Point) -> float:
        # Check the dimensionality of the two points
        dimensionality_check([p1, p2])

        # Set the number of dimensions as a dimensionality of the first one
        # (both point has to have the same at this point)
        n_dims = p1.dimensionality

        # Calculate the differences between each of the dimensions
        diffs = [p2.coords[dim] - p1.coords[dim] for dim in range(n_dims)]

        # Return a square root of the difference squares sum
        return sum(diff ** 2 for diff in diffs) ** 0.5


class Taxicab(Metric):
    """Taxicab (also called Manhattan) distance is responsible for calculating
    the distance as a sum of absolute differences between two points across
    all of the dimensions.

    It also can be represented as a number of steps needed to be performed in
    the orthogonal cartesian multidimensional space.
    """

    def distance(self, p1: Point, p2: Point) -> float:
        # Check the dimensionality of the two points
        dimensionality_check([p1, p2])

        # Set the number of dimensions as a dimensionality of the first one
        # (both point has to have the same at this point)
        n_dims = p1.dimensionality

        # Calculate the absolute differences between each of the dimensions
        abs_diffs = [abs(p2.coords[d] - p1.coords[d]) for d in range(n_dims)]

        # Return sum of these differences
        return sum(abs_diffs)


class Hamming(Metric):
    """Hamming metric is one of the simplest one distances. It simply evaluates
    the number of differences in the given sets.

    In the case of the points in a multidimensional space, it tries to count
    all of the dimensions the two point have different values. For this reason
    it returns an integer instead of more common float (real number).
    """

    def distance(self, p1: Point, p2: Point) -> int:
        # Check the dimensionality of the two points
        dimensionality_check([p1, p2])

        # Set the number of dimensions as a dimensionality of the first one
        # (both point has to have the same at this point)
        n_dims = p1.dimensionality

        # Counter of the differences
        differences = 0

        # Count different values of coordinates across all the dimensions
        for dimension in range(n_dims):
            if p1.coords[dimension] != p2.coords[dimension]:
                differences += 1

        return differences
