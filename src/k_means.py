"""This module contains the most basic implementation of the K-Means algorithm.
"""

from typing import Iterable
from random import sample

from src.datapoint import Centroid, Point
from src.metric import Metric, Euclidean


class KMeans:
    """K-Means algorithm is so called Centroid-based algorithm, that tries to
    iteratively find the centers of potential clusters, while it's trying to
    minimize the distances between the given datapoints and the actual centers.

    This object-based implementation is actually far from being used in
    production environments and should be used just for testing and educational
    purposes.
    """

    def __init__(self,
                 n_clusters: int,
                 metric: Metric = Euclidean(),
                 accuracy: float = 0.001):
        """Initor of the class taking the input parameters under which the
        algorithm will be searching for the knowledge.

        It takes the given number of clusters that should be searched for,
        optional metric used to calculate the mutual distance between two
        points (euclidean distance by default) and an optional parameter
        for accuracy to distinguish the acceptable level of error. This is
        just a security feature to make the algorithm more reliable and not
        to get stuck in the local minima or to prevent kind of infinite
        oscillation between possible solutions that might be sufficient.
        """
        self.__n_clusters = n_clusters
        self.__centroids: list[Centroid] = []
        self.__metric = metric
        self.__accuracy = accuracy

    @property
    def number_of_clusters(self) -> int:
        """Number of wanted clusters. This is the equivalent to the `k`
        variable from the algorithm name."""
        return self.__n_clusters

    @property
    def centroids(self) -> tuple[Centroid]:
        """Calculated centroids. When the model haven't been trained yet,
        it returns an empty tuple."""
        return tuple(self.__centroids)

    @property
    def metric(self) -> Metric:
        """Metric used for training. By default, it's the euclidean distance.
        """
        return self.__metric

    @property
    def accuracy(self) -> float:
        """Acceptable level of error. This some kind represents the sufficiency
        of the found solution."""
        return self.__accuracy

    def closest_centroid(self, p: Point) -> Centroid:
        """This function tries to find the closest centroid in one of the
        already found."""
        if not self.centroids:
            raise KMeansError("Model haven't been trained yet")
        return min(self.centroids, key=lambda c: self.metric.distance(c, p))

    def centroid_by_name(self, name: str) -> Centroid:
        """Tries to find a centroid by given name. If there is no such
        centroid, it returns `None`."""
        for centroid in self.centroids:
            if centroid.name == name:
                return centroid

    def has_centroid(self, name: str) -> bool:
        """Returns if the model contains a cluster called this way."""
        return self.centroid_by_name(name) is not None

    def train(self, points: Iterable[Point], defaults: Iterable[Point] = ()):
        """The base method of the whole class responsible for training the
        model by given data.

        It takes the iterable of datapoints, that should be processed by the
        cluster analysis and an optional interable of default points to be
        taken as default centroids.

        The actual algorithm is based on iteratively performing this actions:

            1. For every given point, assign it to the closest centroid
            2. For every cluster, recalculate the actual centroid
            3. If the position of any of the centroids changed, go to step 1.
        """
        # Initialize the values and process them into wanted form
        self.__centroids = list(defaults)
        points = tuple(points)

        # If no default centroids were provided, randomly select some
        # of the given points
        if not self.__centroids:
            for point in sample(population=points, k=self.number_of_clusters):
                self.__centroids.append(Centroid.centroid_of_point(point))

        # Check the number of given default clusters; if it is not
        # the same as wanted number of clusters (defined in the initor)
        elif len(self.__centroids) != self.number_of_clusters:
            raise KMeansError("Inconsistent number of clusters")

        # Indicator of centroid movement
        changed = True

        # While any of the centroids have changed its coordinates,
        # repeat another iteration
        while changed:
            changed = False

            # Flush all the assigned instances of the centroid
            for centroid in self.centroids:
                centroid.flush()

            # For every given point, assign it to the closes centroid
            for point in points:
                self.closest_centroid(point).add_point(point)

            # Recalculate new centroids for the recently assigned datapoints
            recalc = [centroid.recalculate() for centroid in self.__centroids]

            # Check if any of the centroids have moved for distance larger,
            # than the defined accuracy level (defined in the initor).
            # If any have changed its position for greater distance, change
            # the flag to again (repeat the whole while loop again)
            for i, new_center in enumerate(recalc):
                dist = self.metric.distance(new_center, self.__centroids[i])
                if dist > self.accuracy:
                    changed = True
                    break

            # Set the recently recalculated centroids
            self.__centroids = recalc


class KMeansError(Exception):
    """This exception represents error that may occur when trying to make or
    to train a model but it runs into some inconsistent state."""

    def __init__(self, message: str):
        """Initor taking the message about the error and passing it to the
        parent class' initor."""
        super().__init__(message)

