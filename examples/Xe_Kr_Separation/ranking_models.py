from typing import Sequence, Iterator

import numpy as np
from numpy.typing import NDArray

import ami.abc
from ami.abc import SchemaInterface, RankerInterface, Feature, Target
from ami.abc.ranker import Index
from ami.schema import Schema


# ---------------------------------------------------------------------------------------


class RandomRanker(RankerInterface):
    """Used to randomly sample at start of screening as initial ranker.
    Could instead use clustering algorithm / other approach to select initial indices for screening.
    """    
    def rank(self, x: Sequence[Feature]) -> Iterator[Index]:
        my_rank = np.arange(len(x), dtype=int)
        np.random.shuffle(my_rank)
        print("Current ranker being used is RANDOM")
        return my_rank

    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        pass

    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('index', int)],
            output_schema=[('target', float)]
        )


# ---------------------------------------------------------------------------------------

class SurrogateModelRanker(ami.abc.RankerInterface):
    
    def __init__(self, model, acquisitor) -> None:
        self.model = model
        self.acquisitor = acquisitor
        
    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        """Fit model to passed data points

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            indices of data points to use.
            
        y_val : NDArray[np.int_]
            taret values of data points.
        """
        self.model.fit(x, y)  
        print(f"The shape of the parsed x_array in SurrogateMondelRanker.fit() are {x.shape}")#
        
    def rank(self, x: Sequence[Feature]) -> Iterator[Index]:
        """Rank the passed indices from highest to lowest.
        Highest ranked are highest recommended to be sampled.

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            _description_

        Returns
        -------
        NDArray[np.float_]
            ranked highest to lowest, element 0 is largest ranked, element -1 is lowest ranked.
        """
        alpha = self.determine_alpha()
        alpha_x = alpha[x]
        rankings = np.argsort(alpha_x)[::-1]
        print("Current ranker being used is GPR")
        return rankings  # index of largest alpha is first
    
    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('index', int)],
            output_schema=[('target', float)]
        )
        

class PosteriorRanker(SurrogateModelRanker):
    
    def __init__(self, model, acquisitor, n_post=100) -> None:
        super().__init__(model, acquisitor)
        self.n_post = int(n_post)
    
    def determine_alpha(self) -> NDArray:
        """Determine the alpha (ranking values) for all entries in the full dataset.

        Parameters
        ----------
        None

        Returns
        -------
        NDArray
            alpha values for each entry in the full dataset, non sorted.
        """
        posterior = self.model.sample_y(n_samples=self.n_post)
        alpha = self.acquisitor.score_points(posterior)
        return alpha


# ---------------------------------------------------------------------------------------

    
class ExpectedImprovementRanker(SurrogateModelRanker):
    
    def __init__(self, model, acquisitor) -> None:
        super().__init__(model, acquisitor)
        self._ymax = 0.0
        
    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        super().fit(x, y)
        self._ymax = np.max(y)
        mu, std = self.model.predict()#
        print(f"The shape of the parsed x_array in ExpectedIm.fit() are {x.shape}")#
        print(f" The top Target so far: {self._ymax}")#
    
    def determine_alpha(self) -> NDArray:
        """Determine the alpha (ranking values) for all entries in the full dataset.

        Parameters
        ----------
        None

        Returns
        -------
        NDArray
            alpha values for each entry in the full dataset, non sorted.
        """
        mu, std = self.model.predict()
        alpha = self.acquisitor.score_points(mu, std, self._ymax)
        return alpha
        
    
# ---------------------------------------------------------------------------------------
