import numpy as np
from pykoopman import observables

custom_obs = observables.CustomObservables(
        observables=[
            # lambda px, py, theta: np.arctan2(np.sin(theta), np.cos(theta)),
            lambda px, py, theta, v: np.sin(theta),
            lambda px, py, theta, v: np.sin(2*theta),
            lambda px, py, theta, v: np.sin(3*theta),
            lambda px, py, theta, v: np.sin(4*theta),
            lambda px, py, theta, v: np.cos(theta),
            lambda px, py, theta, v: np.cos(2*theta),
            lambda px, py, theta, v: np.cos(3*theta),
            lambda px, py, theta, v: np.cos(4*theta),

            lambda px, py, theta, v: px**2,
            lambda px, py, theta, v: py**2,
            lambda px, py, theta, v: px*py,
            lambda px, py, theta, v: py**3,
            lambda px, py, theta, v: px**2 * py,
            lambda px, py, theta, v: py**2 * px,

            lambda px, py, theta, v: px * v,
            lambda px, py, theta, v: py * v,
            lambda px, py, theta, v: v**2,
            lambda px, py, theta, v: v**3,

            lambda px, py, theta, v: px * np.cos(theta),
            lambda px, py, theta, v: px * np.sin(theta),
            lambda px, py, theta, v: py * np.cos(theta),
            lambda px, py, theta, v: py * np.sin(theta),

            lambda px, py, theta, v: v * np.cos(theta),
            lambda px, py, theta, v: v * np.sin(theta),
        ]
    )


