import os
import numpy as np
import pykoopman as pk
from pykoopman import regression
from KRTBInterface import KRTBInterface
from pykoopman import observables as obs
import matplotlib.pyplot as plt


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
    #     npzFile_path = sim_traj_path
    # )
    sim_traj = krtb.read_sim_trajectories(sim_traj_path)
    X, Y = krtb.form_data_snapshots(sim_traj)

    # ----------------- Displays simulated trajectories
    # krtb.plot_trajectories(sim_traj[10: 12], d1=0, d2=1)
    
    # ----------------- fits EDMD model
    EDMD = regression.EDMD()
    model_EDMD = pk.Koopman(
        observables=obs.Polynomial(degree=2, include_bias=False),
        regressor=EDMD
    )
    model_EDMD.fit(X, Y, dt=dt)

    # ----------------- prints eigenvalues
    # for i, eig_v in enumerate(model_EDMD.continuous_lamda_array):
    #     print(f"cont eigenvalue {i} = {eig_v:.2f}")

    # ----------------- check validity of eigenfunctions
    # efun_index, linearity_error = model_EDMD.validity_check(np.arange(0, T, dt), X[:int(T/dt), :])
    # print("Ranking of eigenfunctions by linearity error: ", efun_index)
    # print("Corresponding linearity error: ", linearity_error)

    # ----------------- compare simulated and koopman prediction trajectories
    unseen_sim_traj = krtb.get_sim_trajectories(num_traj = 5, T = 5, deltaT = dt, rand_seed = 11)
    krtb.plot_koopman_sim(model_EDMD, sim_traj=unseen_sim_traj, d1=0, d2=1)


if __name__ == "__main__":
    main()