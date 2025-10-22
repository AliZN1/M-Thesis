import os
import numpy as np
import matplotlib.pyplot as plt


from krtb import (
    simulate_trajectories,
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

    def get_sim_trajectories(self, num_traj, T, deltaT=0.01, rand_seed=None, WriteToFile=False, npzFile_path="", data_var=2):
        self.n_traj = num_traj
        self.n_int = int(T/deltaT) + 1
        if WriteToFile and not npzFile_path:
            raise Exception("To read or write the trajectories data from/to a file, npzFile_path argument must be provided.")
        
        if rand_seed:
            np.random.seed(rand_seed)

        init_x = data_var*np.random.random([num_traj, self.system.dim]) - data_var/2
    
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

    def time_reach_bounds(self, koop_model, valid_ef_inx = np.array([]), n_samples_init = 10, initial_sets=None, n_samples_target = 10, target_sets=None, rand_seed = 7):
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

        # ef_initial, ef_target = self.normalize_eigenfunctions(ef_initial, ef_target)
        # ef_initial, ef_target = self.normalize_eigenfunctions_by_eq(koop_model, ef_initial, ef_target, np.zeros(self.system.dim))

        # compute reach time bounds
        time_intervals, status = compute_reach_time_bounds(ef_initial.T, ef_target.T, eig_values)

        return time_intervals, status
    
    def normalize_eigenfunctions_by_eq(self, koop_model, ef_init, ef_targ, xe):
        psi_eq = koop_model.psi(xe.reshape(-1,1))
        # divide each eigenfunction by its value at equilibrium
        for i in range(ef_init.shape[0]):
            if psi_eq[i, 0] != 0:
                ef_init[i, :] /= psi_eq[i, 0]
                ef_targ[i, :] /= psi_eq[i, 0]
        return ef_init, ef_targ
    
    def normalize_eigenfunctions(self, ef_init, ef_targ):
        # Normalize so that mean |ψ| on initial set = 1 and mean phase = 0
        for i in range(ef_init.shape[0]):
            c = np.mean(ef_init[i, :])
            if c == 0:
                continue
            phase = np.exp(-1j * np.angle(c))
            scale = 1 / np.abs(c)
            ef_init[i, :] *= phase * scale
            ef_targ[i, :] *= phase * scale
        return ef_init, ef_targ
    

    def verify_reachability(self, n_samples, T=None, deltaT=0.01, rand_seed=None, writeToFile=False, readFromFile=False, npzFile_path="",):
        if (writeToFile or readFromFile) and not npzFile_path:
            raise Exception("To read or write the trajectories data from/to a file, npzFile_path argument must be provided.")
        
        if rand_seed:
            np.random.seed(rand_seed)

        initial_sets = self.config["initial_sets"]
        target_sets = self.config["verification"]["target_sets"]
        bounds = target_sets[0]["bounds"]

        initial_samples = sample_sets(initial_sets, n_samples, rand_seed)

        if not readFromFile:
            trajectories, t = simulate_trajectories(
                system=self.system,
                initial_points=initial_samples,
                T=T,
                dt=deltaT
            )
            if writeToFile: np.savez(npzFile_path, trajectories=trajectories)
        else:
            data_file = np.load(npzFile_path)
            trajectories = data_file[data_file.files[0]]


        def isStateInLimit(state):
            for dim, (lower, upper) in enumerate(bounds):
                if state[dim] < lower or state[dim] > upper:
                    return False
            return True

        for traj_idx, traj in enumerate(trajectories):
            for step, state in enumerate(traj):
                if isStateInLimit(state):
                    print(f"Trajectory {traj_idx}: Target set reached at {step * deltaT:.2f}s")
                    break
            else:
                print(f"Trajectory {traj_idx}: Target set not reached in given simulation time.")

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


    def test(self, ef_initial, ef_target, eig_values):
        for i, lam in enumerate(eig_values):
            Ti_arr = np.real((1/lam) * np.log(np.abs(ef_target[i,:]/ef_initial[i,:])))
            print(f"Mode {i}: estimated reach time ≈ [{Ti_arr.min():.3f}, {Ti_arr.max():.3f}]s")