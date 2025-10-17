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

        init_x = 2*np.random.random([num_traj, self.system.dim]) - 1 
    
        trajectories, t = simulate_trajectories(
            system = self.system,
            initial_points = init_x,
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


    def plot_trajectories(self, trajectories, d1=None, d2=None, x_lim=[-2, 2], y_lim=[-2, 2]):
        fig, ax = self._get_phase_portrait(d1=d1, d2=d2, x_lim=[-2, 2], y_lim=[-2, 2])

        for traj in trajectories:
            x1 = traj[:, 0]
            x2 = traj[:, 1]
            plt.plot(x1, x2, "-or")
            
        if self.system.dim > 2:
            ax.set_xlabel(f"x{d1+1}")
            ax.set_ylabel(f"x{d2+1}")
        else:
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

        ax.set_title(f"Trajectory projection")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plt.tight_layout()
        plt.show()

    def _get_phase_portrait(self, d1=None, d2=None, x_lim=[-2, 2], y_lim=[-2, 2]):
        fig, ax = plt.subplots(figsize=(7, 7))

        grid_x = np.linspace(x_lim[0], x_lim[1], 30)
        grid_y = np.linspace(y_lim[0], y_lim[1], 30)
        X, Y = np.meshgrid(grid_x, grid_y)
        points_2d = np.stack([X.ravel(), Y.ravel()], axis=-1)

        if self.system.dim > 2:
            # Create n-dimensional points with zeros for other dimensions
            points_nd = np.zeros((points_2d.shape[0], self.system.dim))
            points_nd[:, d1] = points_2d[:, 0]
            points_nd[:, d2] = points_2d[:, 1]

            vecs = np.array([self.system.ff(*pt) for pt in points_nd])
        elif self.system.dim == 2:
            vecs = np.array([self.system.ff(*pt) for pt in points_2d])
        else:
            raise Exception("trajectories dimensions need to be at least 2")
        
        U = vecs[:, 0].reshape(X.shape)
        V = vecs[:, 1].reshape(X.shape)
        ax.streamplot(X, Y, U, V, color="gray", density=1.0, linewidth=0.5, arrowsize=1)

        return fig, ax

    def plot_koopman_sim(self, koop_model, sim_traj, d1=None, d2=None, x_lim=[-2, 2], y_lim=[-2, 2]):

        fig, ax = self._get_phase_portrait(d1=d1, d2=d2, x_lim=[-2, 2], y_lim=[-2, 2])

        if self.system.dim == 2:
            d1, d2 = 0, 1
        
        for traj in sim_traj:
            koop_traj = koop_model.simulate(traj[0, :], n_steps=traj.shape[0])
            
            ax.plot(traj[:, d1], traj[:, d2], '-or')
            ax.plot(koop_traj[:, d1], koop_traj[:, d2], '-og')

        
        if self.system.dim > 2:
            ax.set_xlabel(f"x{d1+1}")
            ax.set_ylabel(f"x{d2+1}")
        else:
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

        ax.set_title(f"Trajectory projection")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plt.tight_layout()
        plt.show()