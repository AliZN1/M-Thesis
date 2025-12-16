import sys, os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
package_root = project_root / "KRTB_pyKoopman_EDMD"
sys.path.insert(0, str(package_root))

import numpy as np
import matplotlib.pyplot as plt
from ASR import custom_obs
import pykoopman as pk
from pykoopman import observables, regression
from KRTB_pyKoopman import KoopmanModelTrainer, KoopmanAnalysis, integralRK4
from controller import KoopmanMPC
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def plotSimTrajectory(state, control, t):
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))

    ax[0].plot(t, control[:, 0], '-ob', label='v')
    ax[0].plot(t, control[:, 1], '-or', label='w')
    ax[1].plot(state[:, 0], state[:, 1], '--r')

    ax[0].set_xlabel("time s")
    ax[0].set_ylabel("control command")
    ax[0].legend(loc='best')
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    plt.tight_layout()
    plt.show()

def compareKoopmanPrediction(pk_model:pk.Koopman, sim_traj, controls):

    fig, ax = plt.subplots(figsize=(7, 7))

    for states, u in zip(sim_traj, controls):
        pk_traj = pk_model.simulate(states[0, :], u, n_steps=states.shape[0])

        ax.plot(states[:, 0], states[:, 1], '--r')
        ax.plot(pk_traj[:, 0], pk_traj[:, 1], '--g', label="Koopman")

    ax.set_xlabel(f"Pos_x")
    ax.set_ylabel(f"pos_y")

    ax.set_title(f"Koopman Prediction Vs. Simulated Trajectories")
    ax.set_xlim([-3, 2])
    ax.set_ylim([-3, 2])

    plt.tight_layout()
    plt.show()


# def keepThisHere():
#     sample = 22221
#     x = states[sample, :, :]
#     u = controls[sample, :, :]
#     n = x.shape[0]-1

#     x_koop = np.zeros((n, 3))
#     reset_idx = np.arange(0, n, int(n/4))
#     for i in range(1, len(reset_idx)):
#         li, ni = reset_idx[i-1], reset_idx[i]
#         x_koop[li:ni, :] = pk_model.simulate(x[li, :], u=u[li:ni, :], n_steps=ni-li)
#     x_koop[reset_idx[-1]:, :] = pk_model.simulate(x[reset_idx[-1], :], u=u[reset_idx[-1]:, :], n_steps=n-reset_idx[-1])

#     # x_koop = pk_model.simulate(x[0, :], u=u[1:n+1, :], n_steps=n)

#     plt.plot(x[:n, 0], x[:n, 1], '-r')
#     plt.scatter(x_koop[:, 0], x_koop[:, 1])

#     # plt.plot(t_vec[:n], x[:n, 0], '--r')
#     # plt.plot(t_vec[:n], x_koop[:n, 0], '--b')

#     plt.show()

def build_reference(traj, i, Np):
    T = traj.shape[0]
    # Select reference window. Clamp if i+Np exceeds T-1
    end = min(i + Np, T)
    window = traj[i:end]

    # If the window is shorter, repeat the last state
    if window.shape[0] < Np:
        last = window[-1]
        repeat_count = Np - window.shape[0]
        repeat_block = np.tile(last, (repeat_count, 1))
        window = np.vstack((window, repeat_block))

    # Return shape (nx, Np)
    return window.T

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    #--------------------- configurations
    curdir = os.path.dirname(__file__)
    config_path = os.path.join(curdir, "config", "benchmark_ASR_forward.json")
    stream = print

    use_saved_data = True

    # ----------------- generates simulated trajectories and train pyKoopman model
    kmt = KoopmanModelTrainer(config_path, curdir, save_model=True, save_sim_data=True, stream=stream)

    if use_saved_data:
        states, controls, t_vec = kmt.loadSimTrajectories(control=True)
        pk_model = kmt.loadPyKoopmanModel()
    else:
        states, controls, t_vec = kmt.simTrajWithControl(seed=4)
        
        reg = regression.EDMDc()
        obs = custom_obs
        pk_model = kmt.trainPyKoopmanModel(states, controls=controls, regressor=reg, observables=obs)
    
    # ----------------- plot results for comparison
    # display an example of simulated trajectories
    # example_traj = 2011
    # plotSimTrajectory(states[example_traj, :, :], controls[example_traj, :, :], t_vec)



    # ----------------- test KMPC
    traj = states[5] # test trajectory
    # traj = np.zeros((250, 3))
    # traj[:, 0] = np.linspace(0, 5, 250)
    # traj[:, 1] = traj[:, 0]
    # traj[:, 2] = np.full(250, np.pi/4)
    
    Q = np.diag([1.0, 1.0, 10.0, 1.0])
    R = np.diag([0.01, 0.22])
    kmpc = KoopmanMPC(pk_model, R, Q, u_min=np.array([0, -np.pi/6]), u_max=np.array([2, np.pi/6]), n_horizon=10)
    
    dt = 0.05
    px = []
    py = []
    x = traj[0, :]
    for i in range(traj.shape[0]):
        x_ref = build_reference(traj, i, kmpc.Np)
        u = kmpc.run(x, x_ref)
        print(u)
        x = integralRK4(kmt.system.ff, x.reshape(1, -1), u, dt).flatten()
        px.append(x[0])
        py.append(x[1])


    plt.scatter(px, py, label="MPC")
    plt.plot(traj[:, 0], traj[:, 1], '--r', label="True")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()