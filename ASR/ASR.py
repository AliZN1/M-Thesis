import numpy as np
from pykoopman import observables

custom_obs = observables.CustomObservables(
        observables=[
            # lambda px, py, theta: np.arctan2(np.sin(theta), np.cos(theta)),
            lambda px, py, theta: np.sin(theta),
            lambda px, py, theta: np.sin(2*theta),
            lambda px, py, theta: np.sin(3*theta),
            lambda px, py, theta: np.sin(4*theta),
            lambda px, py, theta: np.cos(theta),
            lambda px, py, theta: np.cos(2*theta),
            lambda px, py, theta: np.cos(3*theta),
            lambda px, py, theta: np.cos(4*theta),

            lambda px, py, theta: px**2,
            lambda px, py, theta: py**2,
            lambda px, py, theta: px*py,
            lambda px, py, theta: py**3,
            lambda px, py, theta: px**2 * py,
            lambda px, py, theta: py**2 * px,

            lambda px, py, theta: px * np.cos(theta),
            lambda px, py, theta: px * np.sin(theta),
            lambda px, py, theta: py * np.cos(theta),
            lambda px, py, theta: py * np.sin(theta),
        ]
    )


