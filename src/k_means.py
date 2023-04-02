""""""

from typing import Iterable
from random import sample

from src.datapoint import Centroid, Point
from src.metric import Metric, Euclidean


class KMeans:
    """"""

    def __init__(self,
                 n_clusters: int,
                 metric: Metric = Euclidean(),
                 accuracy: float = 0.01):
        self.__n_clusters = n_clusters
        self.__centroids: list[Centroid] = []
        self.__metric = metric
        self.__accuracy = accuracy

    @property
    def number_of_clusters(self) -> int:
        return self.__n_clusters

    @property
    def centroids(self) -> tuple[Centroid]:
        return tuple(self.__centroids)

    @property
    def metric(self) -> Metric:
        """"""
        return self.__metric

    @property
    def accuracy(self) -> float:
        """"""
        return self.__accuracy

    def closest_centroid(self, p: Point) -> Centroid:
        """"""
        return min(self.centroids, key=lambda c: self.metric.distance(c, p))

    def centroid_by_name(self, name: str) -> Centroid:
        for centroid in self.centroids:
            if centroid.name == name:
                return centroid

    def has_centroid(self, name: str) -> bool:
        return self.centroid_by_name(name) is not None

    def train(self, points: Iterable[Point], defaults: Iterable[Point] = ()):

        self.__centroids = list(defaults)
        points = tuple(points)

        if not self.__centroids:
            for point in sample(population=points, k=self.number_of_clusters):
                self.__centroids.append(Centroid.centroid_of_point(point))

        elif len(self.__centroids) != self.number_of_clusters:
            raise Exception("Inconsistent number of clusters")

        changed = True

        while changed:
            changed = False

            for centroid in self.centroids:
                centroid.flush()

            for p in points:
                self.closest_centroid(p).add_point(p)

            recalc = [centroid.recalculate() for centroid in self.__centroids]

            for i, new_center in enumerate(recalc):
                dist = self.metric.distance(new_center, self.__centroids[i])
                if dist > self.accuracy:
                    changed = True
                    break

            self.__centroids = recalc


class KMeansError(Exception):
    """This exception represents error that may occur when trying to make or
    to train a model but it runs into some inconsistent state."""

    def __init__(self, message: str):
        """Initor taking the message about the error and passing it to the
        parent class' initor."""
        super().__init__(message)

