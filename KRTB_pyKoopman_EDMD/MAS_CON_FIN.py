import os
from datetime import datetime 
import numpy as np
import pykoopman as pk
from pykoopman import regression, observables
from KRTB_pyKoopman import KoopmanModelTrainer, KoopmanAnalysis, GenerateReport
from KRTB_pyKoopman import KoopmanPlot as kp
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    # -----------------  configuration
    curdir = os.path.dirname(__file__)
    config_path = os.path.join(os.path.dirname(__file__), "configs", "benchmark_4AS_CON_FIN_REA_BOX.json")
    report_path = os.path.join(curdir, "reports", "4AS_CON_FIN_REA_BOX.txt")

    report = GenerateReport(report_path)
    report.append(f"Report Date & Time: {datetime.now()}\n\n")

    use_saved_data = True
    stream = report.append

    # ----------------- generates simulated trajectories and train pyKoopman model
    kmt = KoopmanModelTrainer(config_path, curdir, save_sim_data=True, save_model=True, stream=stream)

    if use_saved_data:
        sim_traj, t_vec = kmt.loadSimTrajectories()
        pk_model = kmt.loadPyKoopmanModel()
    else:
        sim_traj, t_vec = kmt.simulateTrajectories(seed=22)
        reg = regression.EDMD()
        obs = observables.Polynomial(degree=3, include_bias=True)
        pk_model = kmt.trainPyKoopmanModel(sim_traj, reg, obs)
    
    # ----------------- plot results for comparison
    kp.plot_trajectories(kmt.system , sim_traj[100: 110], d1=6, d2=7, x_lim=[-3, 3], y_lim=[-3, 3])
    return

    # compare simulated and koopman prediction trajectories
    # unseen_sim_traj, _ = kmt.simulateTrajectoriesCustom(num_traj=5, T=10, dt=0.1, seed=13)
    # kp.compareKoopmanPrediction(kmt.system, pk_model, sim_traj=unseen_sim_traj, d1=0, d2=1)

    # ----------------- analyse the reachability of sets with simulation
    initial_sets = np.array([
            [2.27569841, 2.47569841], [-1.22870703, -1.02870703],
            [-1.75449468, -1.55449468], [1.78350465, 1.98350465],
            [1.94591232, 2.14591232], [1.62672579, 1.82672579],
            [-1.77641671, -1.57641671], [-2.42235598,-2.22235598]
        ])
    target_sets = np.array([
            [-0.1, 0.1],[-0.1, 0.1],
            [-0.1, 0.1], [-0.1, 0.1],
            [-0.1, 0.1], [-0.1, 0.1],
            [-0.1, 0.1], [-0.1, 0.1]
        ])
    # kp.reachabilityWithSimulationMultiD(kmt.system, initial_sets, target_sets, 
    #                               T=7, dt=0.01, x_lim=[-3, 3], y_lim=[-3, 3])

    # ----------------- analyse estimated koopman operator
    kAnalysis = KoopmanAnalysis(config_path, pk_model, stream=stream)

    kAnalysis.listEigVal()
    # _, valid_inx = kAnalysis.eigfunLinearityErr(sim_traj, t_vec, err_threshold=8)
    # res = kAnalysis.koopmanResidual(sim_traj)

    valid_inx = np.array([0, 13, 14, 19, 18, 25, 26, 80, 81, 87, 88, 85, 86])
    kAnalysis.timeReachBound(valid_ef_inx=valid_inx, n_samples=1000, rand_seed=11)
    kAnalysis.verifyReachabilityWithSim(n_samples=100, final_time=7, deltaT=0.01, plot=True)

    report.export()

if __name__ == "__main__":
    main()