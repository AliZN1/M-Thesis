import os
import numpy as np
import pykoopman as pk
from pykoopman import regression
from KRTBInterface import KRTBInterface
from pykoopman import observables as obs
import matplotlib.pyplot as plt


def ex5_plot_principal_eigenfun(model_EDMD, psi1_ind, psi2_ind):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    grid_x = np.linspace(-2, 2, 30)
    grid_y = np.linspace(-2, 2, 30)
    X_1, X_2 = np.meshgrid(grid_x, grid_y)

    psi1_t = X_1**2 + 2*X_2 + X_2**3
    psi2_t = X_1 + np.sin(X_2) + X_1**3
    
    axs[0].scatter(psi1_t, psi2_t, c="blue", s=4)

    axs[0].set_xlabel('ψ₁_true')
    axs[0].set_ylabel('ψ₂_true')
    axs[0].set_title('Analytical ψ₁ and ψ₂')

    Psi = model_EDMD.psi(np.vstack((X_1.ravel(), X_2.ravel())))
    psi1_edmd = Psi[psi1_ind, :].reshape((30, 30))
    psi2_edmd = -1 * Psi[psi2_ind, :].reshape((30, 30))

    err = np.real(((psi1_t - psi1_edmd)**2 + (psi2_t - psi2_edmd)**2)**0.5)
    max_err = np.max(err)
    err = err/max_err
    print(f"Maximum error: {max_err}")

    sc = axs[1].scatter(psi1_edmd, psi2_edmd, cmap='plasma', c=err, s=4)
    cbar = fig.colorbar(sc, ax=axs[1], shrink=0.5)
    cbar.set_label('Normalized error')

    axs[1].set_xlabel('ψ₁_edmd')
    axs[1].set_ylabel('ψ₂_edmd')
    axs[1].set_title('Estimated ψ₁ and ψ₂ using pyKoopman')

    plt.tight_layout()
    plt.show()


def main():
    # -----------------  configuration
    config_path = os.path.join(os.path.dirname(__file__), "configs", "benchmark_NL_EIG_FIN_BACKWARD_REA_BOX.json")
    sim_traj_path = os.path.join(os.path.dirname(__file__), "sim_trajectories", "NL_EIG_FIN.npz")

    T, dt = 2, 0.01

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
    # krtb.plot_trajectories(sim_traj)

    # ----------------- fits EDMD model
    EDMD = regression.EDMD()
    model_EDMD = pk.Koopman(
        observables=obs.Polynomial(degree=3, include_bias=False),
        regressor=EDMD
    )
    model_EDMD.fit(X, Y, dt=dt)

    # ----------------- prints eigenvalues
    # for i, eig_v in enumerate(model_EDMD.continuous_lamda_array):
    #     print(f"cont eigenvalue {i} = {eig_v:.2f}")
    
    # ----------------- plots analytical and estimated principal eigenfunctions 
    # ex5_plot_principal_eigenfun(model_EDMD, psi1_ind=0, psi2_ind=7)

    # ----------------- check validity of eigenfunctions
    # efun_index, linearity_error = model_EDMD.validity_check(np.arange(0, T, dt), X[:200, :])
    # print("Ranking of eigenfunctions by linearity error: ", efun_index)
    # print("Corresponding linearity error: ", linearity_error)

    # ----------------- compare simulated and koopman prediction trajectories
    unseen_sim_traj = krtb.get_sim_trajectories(num_traj = 2, T = 1, deltaT = dt, rand_seed = 11)
    krtb.plot_koopman_sim(model_EDMD, sim_traj=unseen_sim_traj)

if __name__ == "__main__":
    main()