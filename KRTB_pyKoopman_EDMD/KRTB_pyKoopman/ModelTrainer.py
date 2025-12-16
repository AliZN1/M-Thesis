import os, warnings, time
import os.path as path
import numpy as np
import pykoopman as pk
import dill as pickle
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
    trajectories = sol.reshape(steps + 1, pts.shape[0], pts.shape[1] )  # shape: (steps+1, N, dim)

    return trajectories.transpose(1, 0, 2), t

def integralRK4(ff, x, u, dt):
    k1 = ff(x, u).T
    k2 = ff(x + 0.5 * dt * k1, u).T
    k3 = ff(x + 0.5 * dt * k2, u).T
    k4 = ff(x + dt * k3, u).T

    return x + dt * (k1 + 2*k2 + 2*k3 + k4)/6

def formDataSnapshots(sim_traj, controls=None): 
    """sim_traj(num_trajectories, num_integral_steps, num_dimension)"""

    n_traj, n_steps, dim = sim_traj.shape
    n_steps -= 1 

    X = np.zeros((n_steps * n_traj, dim))
    Y = np.zeros_like(X)

    if controls is not None:
        U = np.zeros((n_steps * n_traj, controls.shape[2]))
        for i, (traj, con) in enumerate(zip(sim_traj, controls)):
            s = slice(i*n_steps, (i+1)*n_steps)
            X[s, :] = traj[:-1, :]
            Y[s, :] = traj[1:, :]
            U[s, :] = con[:-1, :]

        return X, Y, U

    for i, traj in enumerate(sim_traj):
        s = slice(i*n_steps, (i+1)*n_steps)
        X[s, :] = traj[:-1, :]
        Y[s, :] = traj[1:, :]

    return X, Y

def sample_set(set, n_sample):
    x_init = np.zeros((n_sample, len(set)))
    
    for i, bounds in enumerate(set):
        x_init[:, i] = np.random.uniform(*bounds, size=n_sample)

    return x_init

# def genControlCmd(n_sample, t_vec, v_scale=None):
#     # gama_lim = (-np.pi/6, np.pi/6)
#     # v_amp = np.random.uniform(*v_scale, size=n_sample)
#     # v_freq = 0.5*np.pi * np.random.rand(n_sample)
#     # gama_amp = np.random.uniform(*gama_lim, size=n_sample)
#     # gama_freq = 2*np.pi * np.random.rand(n_sample)
#     # gama = gama_amp[:, np.newaxis] * np.sin(np.outer(gama_freq, t_vec))

#     v = np.random.uniform(0.05, 0.5, size=(n_sample, 1)) * np.ones((n_sample, t_vec.shape[0]))
#     gama = np.random.uniform(0.1, 0.3, size=(n_sample, 1)) * np.ones((n_sample, t_vec.shape[0]))
#     controls = np.stack((v, gama), axis=-1)

#     return controls


class KoopmanModelTrainer:
    def __init__(self, config_path, dirPath, save_model=False, save_sim_data=False, stream=None):
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
        x_init = sample_set(self.config["domain_bounds"], num_traj)
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
            bounds_str = ""
            for i, bounds in enumerate(self.config["domain_bounds"]):
                bounds_str += f"  x{i+1}: {bounds}\n"

            self.stream(f"Generating Simulated Trajectory to Train a Koopman Model")
            self.stream(f" Num. simulated trajectories: {num_traj}\n Random seed: {seed}\n State bounds: \n{bounds_str}")
            self.stream(f" Simulation time span (sec): {(T_end - T_0):.3f} [{T_0:.3f}, {T_end:.3f}] \n Time step(sec): {dt} \n Process runtime (sec): {simulate.dur:.3f}")
            if self._save_sim_data: self.stream(f" Simulated trajectories are stored at: {self._sim_data_path}")
            self.stream("="*60)
        
        return sim_traj, t
    
    def simTrajOpenLoop(self, genControlCmd, seed=None, num_traj=None, T_0=None, T_end=None, dt=None):
        if seed:
            np.random.seed(seed)
        if num_traj is not None and T_0 is not None and T_end is not None and dt is not None:
            pass
        else:
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

        t_vec = np.arange(T_0, T_end, dt)
        num_steps = len(t_vec)

        print("Simulating trajectories ...")
        @timer
        def sim():
            controls = genControlCmd(num_traj, t_vec)
            sim_traj = np.zeros((num_traj, num_steps, self.system.dim))
            sim_traj[:, 0, :] = x_init
            
            for i in range(1, num_steps):
                x = sim_traj[:, i-1, :]
                u = controls[:, i-1, :]
                sim_traj[:, i, :] = integralRK4(self.system.ff, x, u, dt)
        
            return sim_traj, controls

        sim_traj, controls = sim()
        print("Simulation is Done!")

        if self._save_sim_data:
            # save the simulated trajectories inside a file
            mkdir(self._sim_data_path)
            np.savez(self._sim_data_path, sim_traj, t_vec, controls)

        if self.stream:
            self.stream(f"Generating Simulated Trajectory to Train a Koopman Model")
            self.stream(f" Num. simulated trajectories: {num_traj}\n Random seed: {seed}\n State bounds: \n{bounds_str}")
            self.stream(f" Simulation time span (sec): {(T_end - T_0):.3f} [{T_0:.3f}, {T_end:.3f}] \n Time step(sec): {dt} \n Process runtime (sec): {sim.dur:.3f}")
            if self._save_sim_data: self.stream(f" Simulated trajectories are stored at: {self._sim_data_path}")
            self.stream("="*60)
        
        return sim_traj, controls, t_vec
    
    def simTrajClosedLoop(self, controller, seed=None, num_traj=None, T_0=None, T_end=None, dt=None, **kwargs):
        if seed:
            np.random.seed(seed)
        
        sim_config = self.config["simulation"]
        T_0 = T_0 if T_0 is not None else sim_config["start_time"]
        T_end = T_end if T_end is not None else sim_config["final_time"]
        dt = dt if dt is not None else sim_config["time_step"]
        num_traj = num_traj if num_traj is not None else sim_config["num_sim_trajectories"]

        
        # Simulate trajectories based on configuration stored in config file
        x_init = np.zeros((num_traj, self.system.dim))
        bounds_str = ""
        for i, bounds in enumerate(self.config["domain_bounds"]):
            x_init[:, i] = np.random.uniform(*bounds, size=num_traj)
            bounds_str += f"  x{i+1}: {bounds}\n"

        t_vec = np.arange(T_0, T_end, dt)
        num_steps = len(t_vec)

        print("Simulating trajectories ...")
        @timer
        def sim():
            sim_traj = np.zeros((num_traj, num_steps, self.system.dim))
            sim_traj[:, 0, :] = x_init
            U_used = np.zeros((num_traj, num_steps, self.system.num_u))

            for i in range(1, num_steps):
                x = sim_traj[:, i-1, :]
                u = controller(x, **kwargs)
                U_used[:, i-1, :] = u.T
                sim_traj[:, i, :] = integralRK4(self.system.ff, x, u, dt)
        
            return sim_traj, U_used

        sim_traj, controls = sim()
        print("Simulation is Done!")

        if self.stream:
            self.stream(f"Generating Simulated Trajectory to Train a Koopman Model")
            self.stream(f" Num. simulated trajectories: {num_traj}\n Random seed: {seed}\n State bounds: \n{bounds_str}")
            self.stream(f" Simulation time span (sec): {(T_end - T_0):.3f} [{T_0:.3f}, {T_end:.3f}] \n Time step(sec): {dt} \n Process runtime (sec): {sim.dur:.3f}")
            if self._save_sim_data: self.stream(f" Simulated trajectories are stored at: {self._sim_data_path}")
            self.stream("="*60)
        
        return sim_traj, controls, t_vec
    
    def loadSimTrajectories(self, control=False):
        """Reads simulated trajectories stored in a file"""

        if not path.isfile(self._sim_data_path):
            warnings.warn(f"No data file found with this path: {self._sim_data_path}", RuntimeWarning)
            return None
        
        data_file = np.load(self._sim_data_path)
        sim_traj =  data_file[data_file.files[0]]
        t = data_file[data_file.files[1]]

        if self.stream:
            self.stream(f"Simulated trajectories loaded from: {self._sim_data_path}")
            self.stream(f" Num. simulated trajectories: {sim_traj.shape[0]}")
            self.stream(f"Simulation time span (sec): {(t[-1] - t[0]):.3f} [{t[0]:.3f}, {t[-1]:.3f}]")
            self.stream("="*60)

        if control:
            controls = data_file[data_file.files[2]]
            return sim_traj, controls, t
        else:
            return sim_traj, t
        
            
    
    def trainPyKoopmanModel(self, sim_traj, regressor, observables, controls=None)->pk.Koopman:
        U = None
        if controls is not None:
            X, Y, U = formDataSnapshots(sim_traj, controls)
        else:
            X, Y = formDataSnapshots(sim_traj)
        
        @timer
        def train():
            model = pk.Koopman(observables=observables, regressor=regressor)
            model.fit(X, Y,  u=U, dt=self.config["simulation"]["time_step"])
            return model

        model_EDMD = train()

        if self._save_model:
            # save the simulated trajectories inside a file
            mkdir(self._model_path)
            pickle.dump(model_EDMD, open(self._model_path, 'wb'))

        if self.stream:
            self.stream(f"Training a pyKoopman Model based on Simulated Trajectories\n")
            self.stream(f" Process runtime (sec): {train.dur:.6f}")
            if self._save_model: self.stream(f" Trained model is stored at: {self._sim_data_path}")
            self.stream("="*60)

        return model_EDMD
    
    def loadPyKoopmanModel(self)->pk.Koopman:
        if path.isfile(self._model_path):
            model = pickle.load(open(self._model_path, 'rb'))

            if self.stream:
                self.stream(f"Trained model loaded from: {self._model_path}")
                self.stream("="*60)

            return model
        
        else:
            warnings.warn(f"No model found with this path: {self._model_path}", RuntimeWarning)
            return None
        
    def simulateTrajectoriesCustom(self, num_traj, T, dt, seed=None):
        if seed:
            np.random.seed(seed)
            
        x_init = np.zeros((num_traj, self.system.dim))
        for i, bounds in enumerate(self.config["domain_bounds"]):
            x_init[:, i] = np.random.uniform(*bounds, size=num_traj)

        sim_traj, t = simulate(self.system, x_init, T_min=0, T_max=T, dt=dt)

        return sim_traj, t



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