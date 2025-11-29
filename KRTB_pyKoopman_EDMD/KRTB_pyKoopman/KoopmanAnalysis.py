import time
import numpy as np
from krtb import (
    sample_sets,
    compute_reach_time_bounds,
    load_benchmark_config,
     create_system_from_config
)
from .ModelTrainer import simulate, formDataSnapshots, timer


class KRTBInterface:
    def __init__(self, config_path, pk_model, stream=None):
        self.config = load_benchmark_config(config_path)
        self.system = create_system_from_config(self.config["system"])
        self.pk_model = pk_model

        self.stream = stream

    def timeReachBound(self, valid_ef_inx = np.array([]), n_samples=100, initial_sets=None, target_sets=None, rand_seed=None):
        # sample initial and target sets
        if not initial_sets:
            initial_sets = self.config["initial_sets"]
        if not target_sets:
            target_sets = self.config["verification"]["target_sets"]

        initial_samples = sample_sets(initial_sets, n_samples, rand_seed)
        target_samples = sample_sets(target_sets, n_samples, rand_seed)
        
        if len(initial_samples) == 0 or len(target_samples) == 0:
            raise Exception("✗ No valid samples found!")
        
        # compute eigenfunction values on initial and target sets
        ef_initial = self.pk_model.psi(initial_samples.T)
        ef_target = self.pk_model.psi(target_samples.T)
        
        eig_values = self.pk_model.continuous_lamda_array

        if len(valid_ef_inx) != 0:
            ef_initial = ef_initial[valid_ef_inx, :]
            ef_target = ef_target[valid_ef_inx, :]
            eig_values = eig_values[valid_ef_inx]

        # compute reach time bounds
        @timer
        def compute_rtb():
            return compute_reach_time_bounds(ef_initial.T, ef_target.T, eig_values)
        
        time_intervals, status = compute_rtb()

        res_str = ""
        if len(time_intervals) > 0:
            res_str += "Reach-time bounds (time intervals when trajectory is in target set):\n"
            for i, interval in enumerate(time_intervals):
                if isinstance(interval, (tuple, list)) and len(interval) == 2:
                    t_lower, t_upper = interval
                    res_str += f"  Interval {i+1}: [{t_lower:.6f}, {t_upper:.6f}]\n"
                else:
                    res_str += f"  Interval {i+1}: {interval}\n"
        else:
            res_str += "No reachable time intervals found.\n"


        if self.stream:
            self.stream("Reach-Time bound Analysis\n")
            self.stream(f"Status: {status}")
            self.stream(res_str)
            self.stream(f"Process time: {compute_rtb.dur:.6f}")
            self.stream("=" * 60)

        return time_intervals, status

    def verifyReachabilityWithSim(self, n_samples, final_time, start_time=0, deltaT=0.01, seed=None):
        initial_sets = self.config["initial_sets"]
        target_sets = self.config["verification"]["target_sets"]
        bounds = target_sets[0]["bounds"]

        t_0 = time.time()
        initial_samples = sample_sets(initial_sets, n_samples, seed)
        
        trajectories, _ = simulate(self.system, initial_samples, start_time, final_time, deltaT)
        
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
        t_end = time.time()

        if self.stream:
            self.stream("Result of reachability with simulation\n")
            if np.isnan(time_reach).any():
                self.stream(f"Exist a trajectory that doesn't reach the target set in the given simulation time.")
            else:
                self.stream(f"All trajectories reach the target set with in time bound [{time_reach.min():.3f}, {time_reach.max():.3f}]")

            self.stream(f" Process runtime (sec): {t_end-t_0:.6f}")
            self.stream("="*60)


class KoopmanAnalysis(KRTBInterface):
    def __init__(self, config_path, pk_model, stream=None):
        super().__init__(config_path, pk_model, stream)
        

    def listEigVal(self):
        if not self.stream:
            return 
        
        self.stream("Continuous Eigenvales\n")
        for i, eig_v in enumerate(self.pk_model.continuous_lamda_array):
            self.stream(f" eig_v {i} = {eig_v:.3F}")

        self.stream("="*60)

    def validity_check(self, trajectories, t_vec):
        """Perform a validity check of eigenfunctions.

        The validity check tests the linearity of eigenfunctions phi(x(t)) == phi(x(0))
        * exp(lambda*t).

        Args:
            trajectories: numpy.ndarray, shape (n_trajectories, n_samples, n_input_features)
                State vectors to be checked.
            t_vec: numpy.ndarray, shape (n_samples,)
                Time vector.

        Returns:
            linearity_error: list
                Linearity error for each eigenfunction.
        """

        n_traj, n_step, dim = trajectories.shape
        assert t_vec.shape[0] == n_step
        omega = self.pk_model.continuous_lamda_array # (n_eig, )

        X = trajectories.reshape((n_traj*n_step, dim))
        psi_flat = self.pk_model.psi(X.T) # (n_eig, n_traj*n_step)
        psi = psi_flat.reshape((-1, n_traj, n_step)).transpose(1, 2, 0) #(n_traj, n_step, n_eig)

        exp_term = np.exp(np.outer(t_vec, omega)) #(n_step, n_eig)
        psi_t0 = psi[:, 0, :] #(n_traj, n_eig)
        
        epsilon = 10e-10

        diff = (psi - exp_term[None, :, :] * psi_t0[:, None, :]) / (psi + epsilon)

        # computes error over all time-steps in all given trajectories for each eigenfunction
        err = np.linalg.norm(diff,  axis=1)
        ef_mean_err = np.mean(err, axis=0) 

        return np.argsort(ef_mean_err), ef_mean_err


    def eigfunLinearityErr(self, trajectories, t_vec, err_threshold=None):
        
        @timer
        def computErr():
            return self.validity_check(trajectories, t_vec)

        sorted_index, ef_mean_err = computErr()

        valid_inx = None
        if err_threshold:
            valid_inx = np.where(ef_mean_err < err_threshold)[0]

        if self.stream:
            self.stream("Pykoopman Validity Check\n")
            self.stream(f"Average linearity error per eigenfunction:\n{ef_mean_err}")
            self.stream(f"\nEigenfunction ranking (best to worst):\n{sorted_index}")
            if err_threshold: self.stream(f"\nChosen eigenfunction indices: \n{valid_inx}")

            self.stream(f" Process runtime (sec): {computErr.dur:.6f}")
            self.stream("=" * 60)

        return ef_mean_err, valid_inx

    def koopmanResidual(self, sim_traj):
        X, Y = formDataSnapshots(sim_traj)

        eigVec = self.pk_model._regressor_eigenvectors.T
        eigVal = self.pk_model.lamda_array
            
        phi_X = self.pk_model.observables.transform(X)
        phi_Y = self.pk_model.observables.transform(Y)
        
        res = np.zeros(len(eigVal))
        for i in range(len(eigVal)):
            v = eigVec[:, i]
            lam = eigVal[i]
            res[i] = np.linalg.norm(phi_Y @ v - phi_X @ v * lam)/np.linalg.norm(phi_X @ v)
    
        if self.stream:
            self.stream("Result of Residual Analysis for Associated Eigenfunctions:")
            self.stream(f"Eigenfunction ranking (best to worst):\n{np.argsort(res)}")
            self.stream(f"Residual errors:\n{res}")
            self.stream("=" * 60)

        return res

    def koopmanResidual2(self, sim_traj):
        X, Y = formDataSnapshots(sim_traj)

        eigVec = self.pk_model._regressor_eigenvectors.T
        eigVal = self.pk_model.lamda_array

        phi_X = self.pk_model.observables.transform(X)
        phi_Y = self.pk_model.observables.transform(Y)
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

        if self.stream:
            self.stream("Result of Residual Analysis for Associated Eigenfunctions:")
            self.stream(f"Eigenfunction ranking (best to worst):\n{np.argsort(res)}")
            self.stream(f"Residual errors:\n{res}")
            self.stream("=" * 60)

        return res