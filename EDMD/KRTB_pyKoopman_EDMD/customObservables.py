import numpy as np
from pykoopman import observables as obs


NL_EIG_FIN = obs.CustomObservables(
    observables=[
        lambda x: x,
        lambda x: x**2,
        lambda x, y: x*y,
        lambda x: np.sin(x),
    ], 
    observable_names=[
        lambda s: f"{s}",
        lambda s: f"{s}^2",
        lambda s, t: f"{s}*{t}",
        lambda s: f"sin({s})",
    ]
)