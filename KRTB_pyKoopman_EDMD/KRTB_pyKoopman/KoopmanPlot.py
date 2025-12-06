import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .ModelTrainer import simulate, sample_set


def isStateInLimit(state, bounds):
        for dim, (lower, upper) in enumerate(bounds):
            if lower > upper:
                raise ValueError(f"Invalid interval in dimension {dim}: lower={lower} > upper={upper}")
            
            if state[dim] < lower or state[dim] > upper:
                return False
        return True


def get_phase_portrait(system, d1=None, d2=None, x_lim=[-2, 2], y_lim=[-2, 2]):
    fig, ax = plt.subplots(figsize=(7, 7))

    grid_x = np.linspace(x_lim[0], x_lim[1], 30)
    grid_y = np.linspace(y_lim[0], y_lim[1], 30)
    X, Y = np.meshgrid(grid_x, grid_y)
    points_2d = np.stack([X.ravel(), Y.ravel()], axis=-1)

    if system.dim > 2:
        # Create n-dimensional points with zeros for other dimensions
        points_nd = np.zeros((points_2d.shape[0], system.dim))
        points_nd[:, d1] = points_2d[:, 0]
        points_nd[:, d2] = points_2d[:, 1]

        vecs = np.array([system.ff(*pt) for pt in points_nd])
    elif system.dim == 2:
        vecs = np.array([system.ff(*pt) for pt in points_2d])
    else:
        raise Exception("trajectories dimensions need to be at least 2")
    
    U = vecs[:, 0].reshape(X.shape)
    V = vecs[:, 1].reshape(X.shape)
    ax.streamplot(X, Y, U, V, color="gray", density=1.0, linewidth=0.5, arrowsize=1)

    return fig, ax

def plot_trajectories(system, trajectories, d1=None, d2=None, x_lim=[-2, 2], y_lim=[-2, 2]):
    fig, ax = get_phase_portrait(system, d1=d1, d2=d2, x_lim=x_lim, y_lim=y_lim)

    for traj in trajectories:
        x1 = traj[:, d1]
        x2 = traj[:, d2]
        plt.plot(x1, x2, "-or")
        
    if system.dim > 2:
        ax.set_xlabel(f"x{d1+1}")
        ax.set_ylabel(f"x{d2+1}")
    else:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    ax.set_title(f"Simulated Trajectories")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plt.tight_layout()
    plt.show()

def compareKoopmanPrediction(system, koop_model, sim_traj, d1=None, d2=None, x_lim=[-2, 2], y_lim=[-2, 2]):
    if system.dim == 2:
        d1, d2 = 0, 1

    fig, ax = get_phase_portrait(system, d1=d1, d2=d2, x_lim=x_lim, y_lim=y_lim)
    
    for traj in sim_traj:
        koop_traj = koop_model.simulate(traj[0, :], n_steps=traj.shape[0])
        
        ax.plot(traj[:, d1], traj[:, d2], '-or')
        ax.plot(koop_traj[:, d1], koop_traj[:, d2], '-og')

    
    if system.dim > 2:
        ax.set_xlabel(f"x{d1+1}")
        ax.set_ylabel(f"x{d2+1}")
    else:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    red_proxy = plt.Line2D([0], [0], color='r', marker='o', label='Simulated')
    green_proxy = plt.Line2D([0], [0], color='g', marker='o', label='Koopman Prediction')

    ax.set_title(f"Trajectory projection")
    ax.legend(handles=[red_proxy, green_proxy], loc='upper right')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plt.tight_layout()
    plt.show()

def reachabilityWithSimulation(system, set_init, set_target, T, dt, x_lim=[-3, 3], y_lim=[-3, 3]):
    assert len(set_init) == len(set_target)
    if set_init.ndim == 2:
        set_init = np.expand_dims(set_init, axis=0)
        set_target = np.expand_dims(set_target, axis=0)

    _, ax = get_phase_portrait(system, x_lim=x_lim, y_lim=y_lim)

    for set_idx in range(len(set_init)):
        x_init = sample_set(set_init[set_idx], 10)
        trajectories, _ = simulate(system, x_init, 0, T, dt)
        end_idx = trajectories.shape[1]
        bounds = set_target[set_idx]
        
        for traj in trajectories:
            for i, step in enumerate(traj, bounds):
                if isStateInLimit(step):
                    end_idx = i+1
                    break

            ax.plot(traj[:end_idx, 0], traj[:end_idx, 1], '-g')

    for _set in np.vstack((set_init, set_target)):
        w = _set[0, 1] - _set[0, 0] 
        h = _set[1, 1] - _set[1, 0]
        rect = patches.Rectangle((_set[0, 0], _set[1, 0]), w, h, edgecolor='r', facecolor='none', zorder=3)
        ax.add_patch(rect)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_title("Trajectory Reachability Toward Target Sets")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.tight_layout()
    plt.show()

def reachabilityWithSimulationMultiD(system, set_init, set_target, T, dt, x_lim=[-3, 3], y_lim=[-3, 3]):
    assert len(set_init) == len(set_target)
    
    _, ax = plt.subplots(figsize=(7, 7))

    x_init = sample_set(set_init, 10)
    trajectories, _ = simulate(system, x_init, 0, T, dt)
    end_idx = trajectories.shape[1]
    
    for traj in trajectories:
        for i, step in enumerate(traj):
            if isStateInLimit(step, set_target):
                end_idx = i+1
                break

        for dim in range(0, system.dim, 2):
            ax.plot(traj[:end_idx, dim], traj[:end_idx, dim+1], '-g')
    
    all_sets = np.vstack((set_init, set_target))
    for set_idx in range(0, len(all_sets), 2):
        w = all_sets[set_idx, 1] - all_sets[set_idx, 0] 
        h = all_sets[set_idx+1, 1] - all_sets[set_idx+1, 0] 
        rect = patches.Rectangle((all_sets[set_idx, 0], all_sets[set_idx+1, 0]), w, h, edgecolor='r', facecolor='none', zorder=3)
        ax.add_patch(rect)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_title("Reachability Toward Target Set")
    ax.set_xlabel("x1")
    ax.set_ylabel("y1")

    plt.tight_layout()
    plt.show()