import os
from datetime import datetime
import numpy as np
from pykoopman import regression, observables
from KRTB_pyKoopman import KoopmanModelTrainer, KoopmanAnalysis, GenerateReport
from KRTB_pyKoopman import KoopmanPlot as kp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    psi2_edmd = Psi[psi2_ind, :].reshape(X_1.shape)  # check sign convention

    # Compute errors
    eps = 1e-8
    err1 = np.abs((psi1_t - psi1_edmd) / (psi1_t + eps))
    err2 = np.abs((psi2_t - psi2_edmd) / (psi2_t + eps))
    err = np.maximum(err1, err2)

    sc = axs[1].scatter(psi1_edmd, psi2_edmd, cmap='plasma', c=err, norm=mcolors.LogNorm(vmin=1e-2, vmax=1e3) , s=4)
    cbar = fig.colorbar(sc, ax=axs[1], shrink=0.5)
    cbar.set_label('Error')

    axs[1].set_xlabel('ψ₁_edmd')
    axs[1].set_ylabel('ψ₂_edmd')
    axs[1].set_title('Estimated ψ₁ and ψ₂ using pyKoopman')

    plt.tight_layout()
    plt.show()


def main():
    # -----------------  configuration
    curdir = os.path.dirname(__file__)
    config_path = os.path.join(curdir, "configs", "benchmark_NL_EIG_FIN_BACKWARD_REA_BOX.json")
    report_path = os.path.join(curdir, "reports", "NL_EIG_FIN_BACKWARD_REA_BOX.txt")

    use_saved_data = True
    report = GenerateReport(report_path)
    report.append(f"Report Date & Time: {datetime.now()}\n\n")

    # ----------------- generates simulated trajectories and train pyKoopman model
    kmt = KoopmanModelTrainer(config_path, curdir, save_sim_data=True, save_model=True, stream=report.append)

    if use_saved_data:
        sim_traj, t_vec = kmt.loadSimTrajectories()
        pk_model = kmt.loadPyKoopmanModel()
    else:
        sim_traj, t_vec = kmt.simulateTrajectories(seed=22)
        reg = regression.EDMD()
        obs = observables.Polynomial(degree=7, include_bias=True)
        pk_model = kmt.trainPyKoopmanModel(sim_traj, reg, obs)

    # ----------------- plot results for comparison
    # Displays simulated trajectories
    # kp.plot_trajectories(krtb.system, sim_traj[10001: 10010, :, :])

    # compare simulated and koopman prediction trajectories
    # unseen_sim_traj = krtb.get_sim_trajectories(num_traj = 10, T = 1, deltaT = dt, rand_seed = 19)
    # kp.plot_koopman_sim(krtb.system, model_EDMD, sim_traj=unseen_sim_traj)

    # plots analytical and estimated principal eigenfunctions 
    # ex5_plot_principal_eigenfun(model_EDMD, psi1_ind=25, psi2_ind=30)

    # ----------------- analyse estimated koopman operator
    kAnalysis = KoopmanAnalysis(config_path, pk_model, stream=report.append)

    kAnalysis.listEigVal()
    _, valid_idx = kAnalysis.eigfunLinearityErr(sim_traj, t_vec, err_threshold=1)
    # res = kAnalysis.koopmanResidual(sim_traj)

    kAnalysis.timeReachBound(valid_ef_inx=valid_idx, n_samples=1000)
    kAnalysis.verifyReachabilityWithSim(n_samples=100, final_time=2, deltaT=0.01)

    report.export()

if __name__ == "__main__":
    main()