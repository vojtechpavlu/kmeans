from typing import TYPE_CHECKING

# Problems with circular imports
if TYPE_CHECKING:
    from src.metric import Metric
    from datapoint import (Point,
                           Example,
                           Centroid,
                           InconsistentDimensionalityError,
                           NormalizationError)

    from metric import Metric, Euclidean, Taxicab, Hamming
    from k_means import KMeans, KMeansError
