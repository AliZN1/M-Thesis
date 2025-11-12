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
    config_path = os.path.join(os.path.dirname(__file__), "configs", "benchmark_4AS_CON_FIN_REA_BOX.json")
    sim_traj_path = os.path.join(os.path.dirname(__file__), "sim_trajectories", "4AS_CON_FIN.npz")

    T, dt = 12, 0.1
    # ----------------- generates data snapshots from a nonlinear system
    krtb = KRTBInterface(config_path)
    # sim_traj = krtb.get_sim_trajectories(
    #     num_traj = 10000,
    #     T = T,
    #     deltaT = dt,
    #     rand_seed = 42,
    #     WriteToFile = True,
    #     npzFile_path = sim_traj_path,
    #     data_var=3
    # )
    sim_traj = krtb.read_sim_trajectories(sim_traj_path)

    print("number of simulated trajectories: ", sim_traj.shape[0])
    print("="*60)

    X, Y = krtb.form_data_snapshots(sim_traj)

    # ----------------- Displays simulated trajectories
    # kp.plot_trajectories(krtb.system , sim_traj[101: 102], d1=0, d2=1, x_lim=[-3, 3], y_lim=[-3, 3])
    
    # ----------------- fits EDMD model
    EDMD = regression.EDMD()
    model_EDMD = pk.Koopman(
        observables=obs.Polynomial(degree=3, include_bias=False),
        regressor=EDMD
    )
    model_EDMD.fit(X, Y, dt=dt)

    # ----------------- compare simulated and koopman prediction trajectories
    # unseen_sim_traj = krtb.get_sim_trajectories(num_traj = 5, T = 5, deltaT = dt, rand_seed = 16)
    # kp.plot_koopman_sim(krtb.system, model_EDMD, sim_traj=unseen_sim_traj, d1=0, d2=1)
    # return

    # ----------------- prints eigenvalues
    print("Continuous eigenvales: ")
    for i, eig_v in enumerate(model_EDMD.continuous_lamda_array):
        print(f"cont eigenvalue {i} = {eig_v}")

    print("=" * 60)
    
    # ----------------- check validity of eigenfunctions
    efun_index_mean, mean_err = krtb.check_ef_validity(model_EDMD, sim_traj[:5000 , :, :], T, dt)
    print("Average linearity error per eigenfunction:\n", mean_err)
    print("Eigenfunction ranking (best to worst):\n", efun_index_mean)
    print("=" * 60)
    
    # ----------------- residuals analysis
    # eigVal, res = krtb.koopman_residual(model_EDMD, X, Y)
    # print("Residual analysis:")
    # print("eigenfunction ranking (best to worst):\n", np.argsort(res))
    # print("residual errors:\n", res)
    # print("=" * 60)
    
    # ----------------- time-to-reach bounds
    valid_inx = efun_index_mean[np.where(mean_err < 0.2)[0]]
    print("Chosen eigenfunction indices: \n", valid_inx)

    time_intervals, status = krtb.time_reach_bounds(model_EDMD, valid_ef_inx=valid_inx, n_samples_init=1000, n_samples_target=1000)

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
    print("Result of reachability with simulation: ")
    krtb.verify_reachability(
        n_samples=100,
        T = T,
        deltaT=0.01,
    )

if __name__ == "__main__":
    main()