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
    # kp.plot_trajectories(krtb.system , sim_traj[101: 102], d1=0, d2=1, x_lim=[-3, 3], y_lim=[-3, 3])

    # compare simulated and koopman prediction trajectories
    # unseen_sim_traj = krtb.get_sim_trajectories(num_traj = 5, T = 5, deltaT = dt, rand_seed = 16)
    # kp.plot_koopman_sim(krtb.system, model_EDMD, sim_traj=unseen_sim_traj, d1=0, d2=1)

    # ----------------- analyse estimated koopman operator
    kAnalysis = KoopmanAnalysis(config_path, pk_model, stream=stream)

    kAnalysis.listEigVal()
    _, valid_inx = kAnalysis.eigfunLinearityErr(sim_traj, t_vec, err_threshold=1)
    # res = kAnalysis.koopmanResidual(sim_traj)

    kAnalysis.timeReachBound(valid_ef_inx=valid_inx, n_samples=1000)
    kAnalysis.verifyReachabilityWithSim(n_samples=100, final_time=13, deltaT=0.1)

    report.export()

if __name__ == "__main__":
    main()