import os
import numpy as np
import matplotlib.pyplot as plt


from krtb import (
    simulate_trajectories,
    load_benchmark_config,
    create_system_from_config,
)

class KRTBInterface:
    def __init__(self, config_path):
        self.config = load_benchmark_config(config_path)
        self.system = create_system_from_config(self.config["system"])
        self.n_traj = 0
        self.n_int = 0

    def get_sim_trajectories(self, num_traj, T, deltaT=0.01, rand_seed=None, WriteToFile=False, npzFile_path=""):
        self.n_traj = num_traj
        self.n_int = int(T/deltaT) + 1
        if WriteToFile and not npzFile_path:
            raise Exception("To read or write the trajectories data from/to a file, npzFile_path argument must be provided.")
        
        if rand_seed:
            np.random.seed(rand_seed)

        x = 2*np.random.random([num_traj, self.system.dim]) - 1 
    
        trajectories, t = simulate_trajectories(
            system = self.system,
            initial_points = x,
            T = T,
            dt = deltaT
        )
        if WriteToFile:
            np.savez(npzFile_path, trajectories)

        return trajectories
    
    def read_sim_trajectories(self, npzFile_path):
        data_file = np.load(npzFile_path)
        trajectories = data_file[data_file.files[0]]

        self.n_traj = trajectories.shape[0]
        self.n_int = trajectories.shape[1]
        
        return trajectories
    
    def form_data_snapshots(self, krtb_sim_traj):
        n_steps = krtb_sim_traj.shape[1] - 1
        X = np.zeros((n_steps * self.n_traj, self.system.dim))
        Y = np.zeros_like(X)

        for i, traj in enumerate(krtb_sim_traj):
            s = slice(i*n_steps, (i+1)*n_steps)
            X[s, :] = traj[:-1, :]
            Y[s, :] = traj[1:, :]

        return X, Y

    def plot_trajectories(self, trajectories):
        fig, ax = plt.subplots(figsize=(7, 7))
        x_lim = [-2, 2]
        y_lim = [-2, 2]

        grid_x = np.linspace(x_lim[0], x_lim[1], 30)
        grid_y = np.linspace(y_lim[0], y_lim[1], 30)
        X, Y = np.meshgrid(grid_x, grid_y)
        points = np.stack([X.ravel(), Y.ravel()], axis=-1)
        vecs = np.array([self.system.ff(*pt) for pt in points])
        U = vecs[:, 0].reshape(X.shape)
        V = vecs[:, 1].reshape(X.shape)
        ax.streamplot(X, Y, U, V, color="gray", density=1.0, linewidth=0.5, arrowsize=1)
        
        for traj in trajectories:
            x1 = traj[:, 0]
            x2 = traj[:, 1]
            plt.plot(x1, x2, "-or")

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"plot")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plt.tight_layout()
        plt.show()
