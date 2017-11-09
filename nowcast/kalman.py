
"""
Utility functions for the Kalman filter component of the NowCast
algorithm.

"""

import numpy as np

#: Idealized variance for mod
SIGMA_rain = 1.3  # mm/hr
SIGMA_alpha = 0.05  # unitless
SIGMA_radar = 0.3 #  mm/hr

#: Kalman parameters
H = np.array([[0., 1.,]])
A = np.array([[1., 0.],
              [1., 1.]])
Pk = np.array([[SIGMA_alpha**2, 0.           ],
               [0.            , SIGMA_rain**2]])
# R = np.array([[SIGMA_radar**2, ], [0., ]])
R = SIGMA_radar**2



#: Idealized decay rates for stratiform/convective storms
ALPHA_strat = 0.99
ALPHA_conv = 0.975

#: Thresholds
CLEAR_THR = 0.05  # mm/hr
MIN_MATURE_TREND_THR = 0.06  # mm/hr

#: Markov transition probabilities
P_mature_mature = 0.6
P_mature_decay = 1. - P_mature_mature

def intensity_model(t, beta, alpha):
    """ Assumed exponentially-decaying rain profile.

    .. math::
        I_\mathrm{rain} = \beta\alpha^t,\;t=1,2,\dots,\;\alpha \in [0, 1]

    Parameters
    ----------
    t : int
        Integer "step" of algorithm; e.g. number of sequential alpha-folds of
        the data
    beta : float
        Maximal value of exponential distribution
    alpha : float
        Decay rate of exponential distribution

    """
    return beta * np.power(alpha, t)


def log_intensity_model(t, beta, alpha):
    """ Similar to `intensity_model` but separated component-wise via
    logarithmic transform.

    """
    return np.log(beta) + t*np.log(alpha)
