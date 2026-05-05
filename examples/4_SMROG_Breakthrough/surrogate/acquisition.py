import numpy as np
from scipy.stats import norm


class GreedyNRanking:
    """Determines alpha co-efficient of each entry in `X` by counting the number of times each point
    (from the `n_post` samples taken) appears within the `n_opt`.
    I.e. for all the times Xi in sampled from the posterior, how many times is it in the top n_opt?
    
    References
    ----------
    See arXiv:2009.05418 "Bayesian Screening: Multi-test Bayesian Optimization Applied to in silico
    Material Screening" for further discussion of the method.
    """

    def __init__(self, n_opt=100):
        """
        Parameters
        ----------
        n_opt : int (default = 100)
            Number of points which the Greedy N algorithm is attempting to optimise for.
        """
        self.n_opt = int(n_opt)

    def score_points(self, posterior):
        """Increment the count for each data point the amount of times it appears within the top `n_opt` of the sampled posterior.
        
        Parameters
        ----------
        posterior : Posterior distribution
            Assumes posterior is shaped (len(X), n_posterior_samples)

        Returns
        -------
        NDArray[np.int_]
            counts for the number of times each entry in posterior was in the top `n_opt`
            Shape (len(X), )
        """
        alpha = np.zeros(len(posterior))
        top_n_ind = np.argpartition(posterior.T, -self.n_opt).T[-self.n_opt:]
        v, c = np.unique(top_n_ind, return_counts=True)
        alpha[v] += c
        return alpha
        
    
class EiRanking:
    """Conventional Expected Improvement Ranking.
    Strikes a balance between selecting the next entry in `X` based on the expectation for it to improve 
    upon the previously found largest value.
    """
    
    def score_points(self, mu, std, y_max):
        """
        Parameters
        ----------
        mu : Predicted values for all entries in `X`
            Assumes shape is (len(X), )

        std : Standard deviation for predicted values
            Assumes shape is (len(X), )

        y_max : float
            Largest value experimentally sampled so far

        Returns
        -------
        NDArray[np.float_]
            Shape (len(X), )
        """
        improvement = mu - y_max
        scaled_mu = np.divide(improvement, std)
        alpha = improvement * norm.cdf(scaled_mu) + std * norm.pdf(scaled_mu)
        return alpha