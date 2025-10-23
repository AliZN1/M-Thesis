import os
import numpy as np
import pykoopman as pk
from pykoopman import regression
from KRTB_pyKoopman import KRTBInterface
from KRTB_pyKoopman import KoopmanPlot as kp
from pykoopman import observables as obs
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    # -----------------  configuration
    config_path = os.path.join(os.path.dirname(__file__), "configs", "benchmark_MAS_CON_FIN_REA_BOX.json")
    sim_traj_path = os.path.join(os.path.dirname(__file__), "sim_trajectories", "MAS_CON_FIN.npz")

    T, dt = 11, 0.05

    # ----------------- generates data snapshots from a nonlinear system
    krtb = KRTBInterface(config_path)
    # sim_traj = krtb.get_sim_trajectories(
    #     num_traj = 1000,
    #     T = T,
    #     deltaT = dt,
    #     rand_seed = 42,
    #     WriteToFile = True,
    #     npzFile_path = sim_traj_path,
    #     data_var=3
    # )
    
    sim_traj = krtb.read_sim_trajectories(sim_traj_path)
    X, Y = krtb.form_data_snapshots(sim_traj)

    # ----------------- Displays simulated trajectories
    # kp.plot_trajectories(krtb.system , sim_traj[101: 102], d1=0, d2=1, x_lim=[-3, 3], y_lim=[-3, 3])
    
    # ----------------- fits EDMD model
    EDMD = regression.EDMD()
    model_EDMD = pk.Koopman(
        observables=obs.Polynomial(degree=2, include_bias=False),
        regressor=EDMD
    )
    model_EDMD.fit(X, Y, dt=dt)

    # ----------------- prints eigenvalues
    for i, eig_v in enumerate(model_EDMD.continuous_lamda_array):
        print(f"cont eigenvalue {i} = {eig_v}")
    print("=" * 60)

    # ----------------- compare simulated and koopman prediction trajectories
    # unseen_sim_traj = krtb.get_sim_trajectories(num_traj = 5, T = 5, deltaT = dt, rand_seed = 11)
    # kp.plot_koopman_sim(krtb.system, model_EDMD, sim_traj=unseen_sim_traj, d1=0, d2=1)

    # ----------------- check validity of eigenfunctions
    efun_index_mean, mean_err = krtb.check_ef_validity(model_EDMD, sim_traj[900: , :, :], T, dt)
    print("Average linearity error per eigenfunction:\n", mean_err)
    print("=" * 60)
    print("Eigenfunction ranking (best to worst):\n", efun_index_mean)
    print("=" * 60)

    # ----------------- residuals analysis
    eigVal, res = krtb.residual(model_EDMD, X, Y)
    print(np.log(eigVal)/dt)
    print(np.argsort(res))
    print(res)
    # return
    
    # ----------------- time-to-reach bounds
    time_intervals, status = krtb.time_reach_bounds(model_EDMD, valid_ef_inx=np.array([1]))
    # return
    print(f"Status: {status}")
    if len(time_intervals) > 0:
        print(
            "Reach-time bounds (time intervals when trajectory is in target set):"
        )
        for i, interval in enumerate(time_intervals):
            if isinstance(interval, (tuple, list)) and len(interval) == 2:
                t_lower, t_upper = interval
                print(f"  Interval {i+1}: [{t_lower:.6f}, {t_upper:.6f}]")
            else:
                print(f"  Interval {i+1}: {interval}")
    else:
        print("No reachable time intervals found.")

    print("=" * 60)

    # ----------------- verify reachability with simulation
    # validate_traj_path = os.path.join(os.path.dirname(__file__), "sim_trajectories", "MAS_CON_FIN_verify.npz")
    # krtb.verify_reachability(
    #     n_samples=5,
    #     T = 11,
    #     deltaT=0.05,
    #     readFromFile=True,
    #     npzFile_path=validate_traj_path
    # )

if __name__ == "__main__":
    main()