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

def ex5_plot_principal_eigenfun(model_EDMD, psi1_ind, psi2_ind):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    grid_x = np.linspace(-2, 2, 30)
    grid_y = np.linspace(-2, 2, 30)
    X_1, X_2 = np.meshgrid(grid_x, grid_y)

    # Analytical eigenfunctions
    psi1_t = X_1**2 + 2*X_2 + X_2**3
    psi2_t = X_1 + np.sin(X_2) + X_1**3

    axs[0].scatter(psi1_t, psi2_t, c="blue", s=4)
    axs[0].set_xlabel('ψ₁_true')
    axs[0].set_ylabel('ψ₂_true')
    axs[0].set_title('Analytical ψ₁ and ψ₂')

    # Evaluate EDMD model
    Psi = model_EDMD.psi(np.vstack((X_1.ravel(), X_2.ravel())))
    psi1_edmd = Psi[psi1_ind, :].reshape(X_1.shape)
    psi2_edmd = -Psi[psi2_ind, :].reshape(X_1.shape)  # check sign convention

    # Compute errors
    eps = 1e-8
    err1 = np.abs((psi1_t - psi1_edmd) / (psi1_t + eps))
    err2 = np.abs((psi2_t - psi2_edmd) / (psi2_t + eps))
    err = np.maximum(err1, err2)

    sc = axs[1].scatter(psi1_edmd, psi2_edmd, cmap='plasma', c=err, s=4)
    cbar = fig.colorbar(sc, ax=axs[1], shrink=0.5)
    cbar.set_label('Error')

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
    sim_traj = krtb.get_sim_trajectories(
        num_traj = 1000,
        T = T,
        deltaT = dt,
        rand_seed = 42,
        WriteToFile = True,
        npzFile_path = sim_traj_path
    )
    
    # sim_traj = krtb.read_sim_trajectories(sim_traj_path)
    # return
    X, Y = krtb.form_data_snapshots(sim_traj[:1000, :, :])
    
    # ----------------- Displays simulated trajectories
    # kp.plot_trajectories(krtb.system, sim_traj[10: 12, :, :])

    # ----------------- fits EDMD model
    EDMD = regression.EDMD()
    model_EDMD = pk.Koopman(
        observables=obs.Polynomial(degree=3, include_bias=False),
        regressor=EDMD
    )
    model_EDMD.fit(X, Y, dt=dt)

    # ----------------- prints eigenvalues
    for i, eig_v in enumerate(model_EDMD.continuous_lamda_array):
        print(f"cont eigenvalue {i} = {eig_v:.3F}")
    
    # ----------------- plots analytical and estimated principal eigenfunctions 
    # ex5_plot_principal_eigenfun(model_EDMD, psi1_ind=0, psi2_ind=7)
    
    # ----------------- compare simulated and koopman prediction trajectories
    # unseen_sim_traj = krtb.get_sim_trajectories(num_traj = 2, T = 1, deltaT = dt, rand_seed = 11)
    # krtb.plot_koopman_sim(model_EDMD, sim_traj=unseen_sim_traj)

    # ----------------- check validity of eigenfunctions
    efun_index_mean, mean_err = krtb.check_ef_validity(model_EDMD, sim_traj[900: , :, :], T, dt)
    print("Pykoopman validity check:")
    print("Average linearity error per eigenfunction:\n", mean_err)
    print("Eigenfunction ranking (best to worst):\n", efun_index_mean)
    print("=" * 60)
    
    # ----------------- residuals analysis
    eigVal, res = krtb.residual(model_EDMD, X, Y)
    print("Residual analysis:")
    print("eigenfunction ranking (best to worst):\n", np.argsort(res))
    print("residual errors:\n", res)
    print("=" * 60)
    # return

    # ----------------- time-to-reach bounds
    time_intervals, status = krtb.time_reach_bounds(model_EDMD, valid_ef_inx=np.array([0, 7]))
    
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
    # krtb.verify_reachability(
    #     n_samples=5,
    #     T = T,
    #     deltaT=dt
    # )

if __name__ == "__main__":
    main()