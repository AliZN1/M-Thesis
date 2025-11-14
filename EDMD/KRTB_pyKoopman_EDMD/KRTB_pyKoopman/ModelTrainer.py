import os, joblib, warnings, time
import os.path as path
import numpy as np
import pykoopman as pk
from scipy.integrate import odeint
from krtb import(
     load_benchmark_config,
     create_system_from_config
)


def timer(f):
    """Measures execution time."""
    def wrap(*args, **kwargs):
        t0 = time.time()
        res = f(*args, **kwargs)
        wrap.dur = time.time() - t0

        return res
    return wrap

def mkdir(file_path):
    folder_path = path.split(file_path)[0]
    if not path.isdir(folder_path):
        os.mkdir(folder_path)

@timer
def simulate(dyn, pts, T_min,T_max, dt, round_up=True, forward=True):
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

    return trajectories.transpose(1, 0, 2), t

def formDataSnapshots(sim_traj): 
    """sim_traj(num_trajectories, num_integral_steps, num_dimension)"""

    n_traj, n_steps, dim = sim_traj.shape
    n_steps -= 1 

    X = np.zeros((n_steps * n_traj, dim))
    Y = np.zeros_like(X)
    
    for i, traj in enumerate(sim_traj):
        s = slice(i*n_steps, (i+1)*n_steps)
        X[s, :] = traj[:-1, :]
        Y[s, :] = traj[1:, :]

    return X, Y


class KoopmanModelTrainer:
    def __init__(self, config_path, dirPath, save_model=True, save_sim_data=False, stream=None):
        self.config = load_benchmark_config(config_path)
        self.model_name = self.config['benchmark']['name']
        self.system = create_system_from_config(self.config['system'])

        self.dir_path = dirPath
        self._save_model = save_model
        self._save_sim_data = save_sim_data
        self.stream = stream

        self._model_path = path.join(self.dir_path, 'models', self.model_name+'.pkl')
        self._sim_data_path = path.join(self.dir_path, 'sim_trajectories', self.model_name+'.npz')
    
    def simulateTrajectories(self, seed=None):
        if seed:
            np.random.seed(seed)

        sim_config = self.config["simulation"]
        T_0 = sim_config["start_time"]
        T_end = sim_config["final_time"]
        dt = sim_config["time_step"]
        num_traj = sim_config["num_sim_trajectories"]

        
        # Simulate trajectories based on configuration stored in config file
        x_init = np.zeros((num_traj, self.system.dim))
        bounds_str = ""
        for i, bounds in enumerate(self.config["domain_bounds"]):
            x_init[:, i] = np.random.uniform(*bounds, size=num_traj)
            bounds_str += f"  x{i+1}: {bounds}\n"

        print("Simulating trajectories ...")
        sim_traj, t = simulate(
            dyn = self.system, 
            pts = x_init,
            T_min = T_0, 
            T_max = T_end,
            dt = dt
        )
        print("Simulation is Done!")

        if self._save_sim_data:
            # save the simulated trajectories inside a file
            mkdir(self._sim_data_path)
            np.savez(self._sim_data_path, sim_traj, t)

        if self.stream:
            self.stream(f"Generating Simulated Trajectory to Train a Koopman Model")
            self.stream(f" Num. simulated trajectories: {num_traj}\n Random seed: {seed}\n State bounds: \n{bounds_str}")
            self.stream(f" Simulation time span (sec): {T_end - T_0} [{T_0}, {T_end}] \n Time step(sec): {dt} \n Process runtime (sec): {simulate.dur:.3f}")
            if self._save_sim_data: self.stream(f" Simulated trajectories are stored at: {self._sim_data_path}")
            self.stream("="*60)

        return sim_traj, t
    
    def loadSimTrajectories(self):
        if path.isfile(self._sim_data_path):
            # Read simulated trajectories stored in a file
            data_file = np.load(self._sim_data_path)
            sim_traj =  data_file[data_file.files[0]]
            t = data_file[data_file.files[1]]

            if self.stream:
                self.stream(f"Simulated trajectories loaded from: {self._sim_data_path}")
                self.stream("="*60)

            return sim_traj, t
        
        else:
            warnings.warn(f"No data file found with this path: {self._sim_data_path}", RuntimeWarning)
            return None
    
    def trainPyKoopmanModel(self, sim_traj, regressor, observables)->pk.Koopman:
        X, Y = formDataSnapshots(sim_traj)

        @timer
        def train():
            return pk.Koopman(observables=observables, regressor=regressor)

        model_EDMD = train()
        model_EDMD.fit(X, Y, dt=self.config["simulation"]["time_step"])

        if self._save_model:
            # save the simulated trajectories inside a file
            mkdir(self._model_path)
            joblib.dump(model_EDMD, self._model_path)

        if self.stream:
            self.stream(f"Training a pyKoopman Model based on Simulated Trajectories\n")
            self.stream(f" Process runtime (sec): {train.dur:.6f}")
            if self._save_sim_data: self.stream(f" Trained model is stored at: {self._sim_data_path}")
            self.stream("="*60)

        return model_EDMD
    
    def loadPyKoopmanModel(self)->pk.Koopman:
        if path.isfile(self._model_path):
            model = joblib.load(self._model_path)

            if self.stream:
                self.stream(f"Trained model loaded from: {self._model_path}")
                self.stream("="*60)

            return model
        
        else:
            warnings.warn(f"No model found with this path: {self._model_path}", RuntimeWarning)
            return None
        

class GenerateReport:
    def __init__(self, file_path):
        self.file_path = file_path
        self.context = ""

        if not path.isdir(self.file_path):
            mkdir(self.file_path)

    def append(self, text):
        self.context += (text + "\n")
    
    def clear(self):
        self.context = ""

    def export(self):
        with open(self.file_path, "w") as file:
            file.write(self.context)