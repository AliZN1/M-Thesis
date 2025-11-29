import numpy as np
import matplotlib.pyplot as plt


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
        x1 = traj[:, 0]
        x2 = traj[:, 1]
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