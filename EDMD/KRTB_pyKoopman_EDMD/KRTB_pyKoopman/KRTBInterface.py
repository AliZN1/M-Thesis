import numpy as np
from scipy import linalg
from scipy.integrate import odeint
import math


from krtb import (
    load_benchmark_config,
    create_system_from_config,
    sample_sets,
    compute_reach_time_bounds
)

class KRTBInterface:
    def __init__(self, config_path):
        self.config = load_benchmark_config(config_path)
        self.system = create_system_from_config(self.config["system"])
        self.n_traj = 0
        self.n_int = 0

    
    def simulate(self, dyn, pts, T_min,T_max, dt, round_up=True, forward=True):
        if not isinstance(pts, np.ndarray):
            pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        assert pts.ndim == 2  # N*n matrix, N n-dimensional points
        T = T_max - T_min
        assert T > 0 and T >= dt
        T = np.ceil(T / dt) * dt if round_up else np.floor(T / dt) * dt
        steps = np.ceil(T / dt).astype(int)

        coeff = 1 if forward else -1

        def ode_fx(x, t):
            x = x.reshape(pts.shape)
            y = dyn.ff(*x.T).squeeze(axis=0).T
            return y.flatten() * coeff
        
        # avoid the numerical issue when T_min is 0
        T_min = 1e-6 if T_min == 0 else T_min

        # create the time grid for the entire simulation
        t = np.linspace(T_min, T_max, steps + 1)

        # integrate the entire trajectory in one call
        sol = odeint(ode_fx, pts.flatten(), t)

        # reshape the solution to proper format
        trajectories = sol.reshape(
            steps + 1, pts.shape[0], pts.shape[1]
        )  # shape: (steps+1, N, dim)

        return trajectories.transpose(1, 0, 2)
    
    def get_sim_trajectories(self, num_traj, T, deltaT=0.01, rand_seed=None, WriteToFile=False, npzFile_path="", data_var=2):
        self.n_traj = num_traj
        self.n_int = int(T/deltaT) + 1
        if WriteToFile and not npzFile_path:
            raise Exception("To read or write the trajectories data from/to a file, npzFile_path argument must be provided.")
        
        if rand_seed:
            np.random.seed(rand_seed)

        init_x = data_var*np.random.random([num_traj, self.system.dim]) - data_var/2
        trajectories = self.simulate(self.system, init_x, 0, T, deltaT)

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

    def time_reach_bounds(self, koop_model, valid_ef_inx = np.array([]), n_samples_init = 10, initial_sets=None, target_sets=None, n_samples_target = 10, rand_seed = 7):
        # sample initial and target sets
        if not initial_sets:
            initial_sets = self.config["initial_sets"]
        if not target_sets:
            target_sets = self.config["verification"]["target_sets"]

        initial_samples = sample_sets(initial_sets, n_samples_init, rand_seed)
        target_samples = sample_sets(target_sets, n_samples_target, rand_seed)
        
        if len(initial_samples) == 0 or len(target_samples) == 0:
            raise Exception("✗ No valid samples found!")
        
        # compute eigenfunction values on initial and target sets
        ef_initial = koop_model.psi(initial_samples.T)
        ef_target = koop_model.psi(target_samples.T)
        
        eig_values = koop_model.continuous_lamda_array
    
        if len(valid_ef_inx) != 0:
            ef_initial = ef_initial[valid_ef_inx, :]
            ef_target = ef_target[valid_ef_inx, :]
            eig_values = eig_values[valid_ef_inx]

        # self.test(ef_initial, ef_target, eig_values)

        # compute reach time bounds
        time_intervals, status = compute_reach_time_bounds(ef_initial.T, ef_target.T, eig_values)

        return time_intervals, status

    def verify_reachability(self, n_samples, T=None, deltaT=0.01, rand_seed=None):
        if rand_seed:
            np.random.seed(rand_seed)

        initial_sets = self.config["initial_sets"]
        target_sets = self.config["verification"]["target_sets"]
        bounds = target_sets[0]["bounds"]

        initial_samples = sample_sets(initial_sets, n_samples, rand_seed)
        
        trajectories = self.simulate(self.system, initial_samples, 0, T, deltaT)
        
        def isStateInLimit(state):
            for dim, (lower, upper) in enumerate(bounds):
                if state[dim] < lower or state[dim] > upper:
                    return False
            return True
        
        time_reach = np.full(trajectories.shape[0], np.nan)
        for traj_idx, traj in enumerate(trajectories):
            for step, state in enumerate(traj):
                if isStateInLimit(state):
                    time_reach[traj_idx] = step * deltaT
                    break
            
        if np.isnan(time_reach).any():
            print(f"Exist a trajectory that doesn't reach the target set in the given simulation time.")
        else:
            print(f"All trajectories reach the target set with in time bound [{time_reach.min():.3f}, {time_reach.max():.3f}]")

    @staticmethod
    def check_ef_validity(koop_model, trajectories, T, dt):
        t_vec = np.arange(0, T, dt)
        n_eigs = len(koop_model.continuous_lamda_array)
        n_traj = trajectories.shape[0]

        ef_errors = np.full((n_traj, n_eigs), np.nan)

        for i, traj in enumerate(trajectories):
            efun_index, err = koop_model.validity_check(t_vec, traj[:len(t_vec), :])
            ef_errors[i, efun_index] = err

        ef_mean_err = np.nanmean(ef_errors, axis=0)

        return np.argsort(ef_mean_err), np.sort(ef_mean_err)
    
    @staticmethod
    def residual(koop_model, X, Y, W=None):
        eigVec = koop_model._regressor_eigenvectors.T
        eigVal = koop_model.lamda_array
            
        phi_X = koop_model.observables.transform(X)
        phi_Y = koop_model.observables.transform(Y)
        
        res = np.zeros(len(eigVal))
        for i in range(len(eigVal)):
            v = eigVec[:, i]
            lam = eigVal[i]
            res[i] = np.linalg.norm(phi_Y @ v - phi_X @ v * lam)/np.linalg.norm(phi_X @ v)
        
        return eigVal, res
    
    @staticmethod
    def koopman_residual(koop_model, X, Y):
        eigVec = koop_model._regressor_eigenvectors.T
        eigVal = koop_model.lamda_array

        phi_X = koop_model.observables.transform(X)
        phi_Y = koop_model.observables.transform(Y)
        scale = 1/phi_X.shape[0]

        Gx = scale * (phi_X.conj().T @ phi_X)
        Gy = scale * (phi_Y.conj().T @ phi_Y)
        Gxy = scale * (phi_X.conj().T @ phi_Y)
        Gyx = scale * (phi_Y.conj().T @ phi_X)

        res = np.zeros(len(eigVal))
        for i in range(len(eigVal)):
            g = eigVec[:, i]
            lam = eigVal[i]
            
            M = Gy - lam * Gyx - np.conj(lam) * Gxy + np.abs(lam)**2 * Gx
            num = np.real(np.vdot(g, M @ g))
            den = np.real(np.vdot(g, Gx @ g))
            res[i] = np.sqrt(max(num, 0.0)/max(den, 1e-10))

        return eigVal, res

    def test(self, ef_initial, ef_target, eig_values):
        for i, lam in enumerate(eig_values):
            Ti_arr = np.real((1/lam) * np.log(np.abs(ef_target[i,:]/ef_initial[i,:])))
            print(f"Mode {i}: estimated reach time ≈ [{Ti_arr.min():.3f}, {Ti_arr.max():.3f}]s")