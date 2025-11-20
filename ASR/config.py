import numpy as np
import os
import os.path as path


def formDataSnapshots(sim_traj, controls): 
    """sim_traj(num_trajectories, num_integral_steps, num_dimension)"""

    n_traj, n_steps, dim = sim_traj.shape
    n_steps -= 1 

    X = np.zeros((n_steps * n_traj, dim))
    Y = np.zeros_like(X)
    U = np.zeros((n_steps * n_traj, controls.shape[2]))
    
    for i, (traj, u) in enumerate(zip(sim_traj, controls)):
        s = slice(i*n_steps, (i+1)*n_steps)
        X[s, :] = traj[:-1, :]
        Y[s, :] = traj[1:, :]
        U[s, :] = u[:-1, :]

    return X, Y, U

def mkdir(file_path):
    folder_path = path.split(file_path)[0]
    if not path.isdir(folder_path):
        os.mkdir(folder_path)


class ASRModel:
    def __init__(self, l):
        """ ASRModel constructor
        :param l: length of wheel base
        """
        self.l = l
        self.dim = 3
        self.gama_lim = (-np.pi/6, np.pi/6)
        self.x_lim = [[-1, 1], [-1, 1], [-np.pi, np.pi]]
    
    def ff(self, x, u):
        v = u[:, 0]
        gama = u[:, 1]

        dx = np.zeros_like(x)
        dx[:, 0] = v * np.cos(x[:, 2])
        dx[:, 1] = v * np.sin(x[:, 2])
        dx[:, 2] = v * np.tan(gama) / self.l

        return dx

    def genControlCmd(self, n_sample, t_vec, v_scale=None):

        # v_amp = np.random.uniform(*v_scale, size=n_sample)
        # v_freq = 0.5*np.pi * np.random.rand(n_sample)
        gama_amp = np.random.uniform(*self.gama_lim, size=n_sample)
        gama_freq = 2*np.pi * np.random.rand(n_sample)


        v = np.ones((n_sample, t_vec.shape[0]))
        gama = gama_amp[:, np.newaxis] * np.sin(np.outer(gama_freq, t_vec))
        controls = np.stack((v, gama), axis=-1)

        return controls
    
    def rk4(self, x, u, dt):
        k1 = self.ff(x, u)
        k2 = self.ff(x + 0.5 * dt * k1, u)
        k3 = self.ff(x + 0.5 * dt * k2, u)
        k4 = self.ff(x + dt * k3, u)

        return x + dt * (k1 + 2*k2 + 2*k3 + k4)/6

    def simulate(self, n_samples, t_final, dt, seed=None):
        if seed is not None:
            np.random.seed(seed)

        t_vec = np.arange(0, t_final+dt, dt)
        n_steps = t_vec.shape[0]
        controls = self.genControlCmd(n_samples, t_vec)

        # creates random initial states
        x_init = np.zeros((n_samples, self.dim))
        for i, lim in enumerate(self.x_lim):
            x_init[:, i] = np.random.uniform(*lim, n_samples)

        # integrates
        traj = np.zeros((x_init.shape[0], n_steps, x_init.shape[1]))
        traj[:, 0, :] = x_init

        for i in range(1, n_steps):
            x = traj[:, i-1, :]
            u = controls[:, i-1, :]
            traj[:, i, :] = self.rk4(x, u, dt)
        
        return traj, controls, t_vec