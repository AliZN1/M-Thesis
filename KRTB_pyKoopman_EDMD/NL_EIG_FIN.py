import os
from datetime import datetime
import numpy as np
from pykoopman import regression, observables
from KRTB_pyKoopman import KoopmanModelTrainer, KoopmanAnalysis, GenerateReport
from KRTB_pyKoopman import KoopmanPlot as kp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def estimate_scaling_factor(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()

    return  np.dot(a_flat, b_flat) / np.dot(b_flat, b_flat)

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

    # estimate scaling factor
    c1 = estimate_scaling_factor(psi1_t, psi1_edmd)
    c2 = estimate_scaling_factor(psi2_t, psi2_edmd)
    
    psi1_edmd *= c1
    psi2_edmd *= c2

    # Compute errors
    eps = 1e-8
    err1 = np.abs((psi1_t - psi1_edmd) / (psi1_t + eps))
    err2 = np.abs((psi2_t - psi2_edmd) / (psi2_t + eps))
    err = np.maximum(err1, err2)

    sc = axs[1].scatter(psi1_edmd, psi2_edmd, cmap='plasma', c=err, norm=mcolors.LogNorm(vmin=1e-2, vmax=1e1) , s=4)
    cbar = fig.colorbar(sc, ax=axs[1], shrink=0.5)
    cbar.set_label('error bar')

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

    report = GenerateReport(report_path)
    report.append(f"Report Date & Time: {datetime.now()}\n\n")

    use_saved_data = True
    # stream = report.append
    stream = print

    # ----------------- generates simulated trajectories and train pyKoopman model
    kmt = KoopmanModelTrainer(config_path, curdir, save_sim_data=False, save_model=False, stream=stream)

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
    # rand_num = np.random.randint(0, 10000, 20)
    # kp.plot_trajectories(kmt.system, sim_traj[rand_num, :, :], x_lim=[-1, 3], y_lim=[-2,2])

    # compare simulated and koopman prediction trajectories
    # unseen_sim_traj, _ = kmt.simulateTrajectoriesCustom(num_traj=5, T=1, dt=0.01, seed=11)
    # kp.compareKoopmanPrediction(kmt.system, pk_model, sim_traj=unseen_sim_traj, x_lim=[-3, 5], y_lim=[-3, 3])

    # plots analytical and estimated principal eigenfunctions
    # ex5_plot_principal_eigenfun(pk_model, psi1_ind=33, psi2_ind=22)

    # ----------------- analyse the reachability of sets with simulation
    initial_sets = np.array([
            [[0.0,   0.10], [1.10,   1.20]], # Reachable
            [[0.32,  0.42], [-1.15, -1.25]], # Reachable
            [[0.80,  0.90], [-1.70, -1.60]], # Reachable
            [[-1.0, -0.90], [1.74,   1.84]]  # Unreachable
        ])
    target_sets = np.array([
            [[1.80,   1.90], [-0.80, -0.70]], # Reachable
            [[-2.0,  -1.90], [-1.40, -1.30]], # Reachable
            [[2.87,   2.97], [-1.88, -1.78]], # Reachable
            [[-1.82, -1.72], [0.0,    0.10]]  # Unreachable
        ])
    # kp.reachabilityWithSimulation(kmt.system, initial_sets, target_sets, 
    #                               T=2, dt=0.01, x_lim=[-2.1, 3], y_lim=[-2.1, 3])

    # ----------------- analyse estimated koopman operator
    kAnalysis = KoopmanAnalysis(config_path, pk_model, stream=stream)

    kAnalysis.listEigVal()
    _, valid_idx = kAnalysis.eigFunValidity(sim_traj, t_vec, alpha=0.1, score_threshold=0.012)
    # res = kAnalysis.koopmanResidual(sim_traj)

    kAnalysis.timeReachBound(valid_ef_idx=valid_idx, n_samples=1000)
    kAnalysis.verifyReachabilityWithSim(n_samples=100, final_time=2, deltaT=0.01, plot=False)

    report.export()

if __name__ == "__main__":
    main()