import numpy as np
from scipy.integrate import odeint


class ASRModel:
    def __init__(self, l):
        """ ASRModel constructor
        :param l: length of wheel base
        """
        self.l = l
        self.dim = 4
    
    def ff(self, x, u):
        v = u[:, 0]
        w = u[:, 1]

        dx = np.zeros_like(x)
        dx[:, 0] = v * np.cos(x[:, 2])
        dx[:, 1] = v * np.sin(x[:, 2])
        dx[:, 2] = v * np.tan(x[:, 3]) / self.l
        dx[:, 3] = w

        return dx

    @staticmethod
    def check_control_command(gama, w):
        limit = np.pi / 6
        scale = np.clip(1 - np.abs(gama) / limit, 0, 1)

        return w * scale

    def init_control(self, v_bound, w_scale, n_sample, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.v_amp = np.random.uniform(*v_bound, size=n_sample)
        self.v_freq = 0.5*np.pi * np.random.rand(n_sample)
        self.w_amp = np.random.uniform(*w_scale, size=n_sample)
        self.w_freq = 2*np.pi * np.random.rand(n_sample)

    def control(self, t, gama):
        "Time variant control commands"
        v = self.v_amp * np.cos(self.v_freq * t + np.pi) + self.v_amp
        w = self.w_amp * np.sin(self.w_freq * t)
        w = self.check_control_command(gama, w)

        return np.vstack([v, w]).T

    def simulate(self, x_init, t_final, dt):
        n_sample = x_init.shape[0]
        n_steps = int(t_final/dt) + 1

        controls = np.zeros((n_sample, n_steps, 2))
        t_eval = np.linspace(0, t_final, n_steps)

        def ode_fx(x, t):
            x = x.reshape(x_init.shape) # (n_sample, n_x_dim)
            # compute control commands at odeint specified time
            u = self.control(t, x[:, 3]) #(n_sample, n_u_dim)

            return self.ff(x, u).flatten()

        states = odeint(ode_fx, x_init.flatten(), t_eval)
        states = states.reshape(n_steps, n_sample, self.dim)
        
        # compute control commands at t_eval intervals
        for i, t in enumerate(t_eval):
            controls[:, i, :] = self.control(t, states[i, :, 3])

        return t_eval, states.transpose(1, 0, 2), controls