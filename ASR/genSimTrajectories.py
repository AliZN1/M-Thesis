import matplotlib.pyplot as plt
from model import *
import time


def main():
    np.random.seed(10)

    robot_model = ASRModel(l=0.3)
    n_sample = 10

    # generates random initial states
    pos_init = np.random.uniform(-2, 2, size=(n_sample, 2))
    theta_init = np.random.uniform(-np.pi, np.pi, size=(n_sample, 1))
    gama_init = np.random.uniform(-np.pi/6, np.pi/6, size=(n_sample, 1))
    x_init = np.hstack([pos_init, theta_init, gama_init]) # shape: (n_samples, 4)

    # perform the simulation
    start = time.time()
    robot_model.init_control(v_bound=(1, 2), w_scale=(0.5, 1), n_sample=n_sample, seed=3)
    t, states, controls = robot_model.simulate(x_init, 10, 0.05)
    end = time.time()
    print(f"Simulation took {(end-start):.2f} seconds")

    #display an example of simulated trajectories
    example_traj = 9
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))

    ax[0].plot(t, controls[example_traj, :, 1], '-ob', label='v')
    ax[0].plot(t, controls[example_traj, :, 0], '-or', label='w')
    ax[1].plot(states[example_traj, :, 0], states[example_traj, :, 1], '--r')

    ax[0].set_xlabel("time s")
    ax[0].set_ylabel("control command")
    ax[0].legend(loc='best')
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    plt.show()


if __name__ == "__main__":
    main()