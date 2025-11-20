import numpy as np
import matplotlib.pyplot as plt
from config import ASRModel, formDataSnapshots, mkdir
from controller import LinearMPC
import time
import pykoopman as pk
from pykoopman import observables, regression
import os.path as path
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def compareKoopmanPred(pk_model, robot_model, T, dt, seed=11):
    states, controls, t_vec = robot_model.simulate(1, T, dt, seed=11)

    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    x_koop = pk_model.simulate(states[0, 0, :], controls[0], n_steps=states.shape[1])

    ax[0].plot(t_vec, controls[0, :, 1], '-ob', label='v')
    ax[0].plot(t_vec, controls[0, :, 0], '-or', label='w')
    ax[1].plot(states[0, :, 0], states[0, :, 1], '--r')
    ax[1].plot(x_koop[:, 0], x_koop[:, 1], '--b')

    ax[0].set_xlabel("time s")
    ax[0].set_ylabel("control command")
    ax[0].legend(loc='best')
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    plt.show()


cus_obs = observables.CustomObservables(
    observables=[
        # monomial for x1 and x2
        lambda x1, x2, x3: x1 ** 2,
        lambda x1, x2, x3: x2 ** 2,
        lambda x1, x2, x3: x1 * x2,
        lambda x1, x2, x3: x1 ** 3,
        lambda x1, x2, x3: x2 ** 3,
        lambda x1, x2, x3: x1 ** 2 * x2,
        lambda x1, x2, x3: x2 ** 2 * x1,
        #Fourier for x3
        lambda x1, x2, x3: np.sin(x3),
        lambda x1, x2, x3: np.sin(2*x3),
        lambda x1, x2, x3: np.sin(3*x3),
        lambda x1, x2, x3: np.sin(4*x3),
        lambda x1, x2, x3: np.cos(x3),
        lambda x1, x2, x3: np.cos(2*x3),
        lambda x1, x2, x3: np.cos(3*x3),
        lambda x1, x2, x3: np.cos(4*x3),
    ]   
)


def main():
    dt = 0.02
    robot_model = ASRModel(l=0.3)
    model_path = path.join(path.dirname(__file__), 'model', 'ASR_pk_model.pkl')
    load_model = False

    if not load_model:
        # runs simulation
        n_samples, T = 1000, 10
        start = time.time()
        states, controls, t_vec = robot_model.simulate(n_samples, T, dt, seed=13)
        end = time.time()
        print(f"Simulation took {(end-start):.2f} seconds")

        # train pyKoopman model 
        X, Y, U = formDataSnapshots(states, controls)
        
        reg = regression.EDMDc()
        obs = observables.Polynomial(degree=3)
        obs += observables.RandomFourierFeatures(D=10)

        pk_model = pk.Koopman(observables=cus_obs, regressor=reg)
        pk_model.fit(X, Y, U, dt=dt)

        # mkdir(model_path)
        # joblib.dump(pk_model, model_path)

    else:
        pk_model = joblib.load(model_path)
        print(f"pyKoopman model loaded from file path: {model_path}")

    print(pk_model.get_feature_names())

    #display an example of simulated trajectories
    # compareKoopmanPred(pk_model, robot_model, 20, dt)

    # create reference trajectory
    # x_ref, u_ref, t_ref = robot_model.simulate(1, 15, 0.05, seed=13)
    # x_ref, u_ref = x_ref.squeeze(), u_ref.squeeze()
    # z_ref = pk_model.observables.transform(x_ref).T

    # print(x_ref.shape)
    z_point_ref = pk_model.observables.transform(np.array([[-4, 0, 2.96944038]])).T

    mpc = LinearMPC(A=pk_model.A, B=pk_model.B, 
                    u_min=np.array([0, -np.pi/6]), u_max=np.array([1.5, np.pi/6]))
    
    mpc.Q[0, 0] = 10.0   # weight on x
    mpc.Q[1, 1] = 10.0   # weight on y
    mpc.Q[2, 2] = 5.0    # weight on theta
    
    # z0 = pk_model.observables.transform(x_ref[0].reshape(1, -1)).T
    n_steps = int(15 / 0.05)
    X = []

    z = z0.copy()
    for t in range(n_steps):
        # t_h = np.clip(t+10, a_min=t, a_max=len(t_ref))
        # u_mpc = mpc.run(z, z_ref[:, t:t_h])
        u_mpc = mpc.run(z, z_point_ref)

        z = pk_model.A @ z + pk_model.B @ u_mpc
        X.append(pk_model.C @ z)

    X = np.array(X).squeeze()
    # plt.plot(x_ref[:, 0], x_ref[:, 1], '--b')
    plt.plot(X[:, 0], X[:, 1], '--r')
    plt.show()

if __name__ == "__main__":
    main()