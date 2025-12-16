import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import solve_discrete_are
from pykoopman import observables, regression
from KRTB_pyKoopman import KoopmanModelTrainer, KoopmanAnalysis, integralRK4
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def genControlCmd(n_sample, t_vec): 
    # amp = np.zeros(n_sample) 
    amp = np.random.uniform(-1, 1, size=n_sample) 
    f = 0.5 # signal frequency (Hz) 
    U = np.outer(amp, (np.sin(2 * np.pi * f * t_vec) >= 0)) 
    return np.expand_dims(U, axis=2)

def main():
    curdir = os.path.dirname(__file__)
    config_path = os.path.join(curdir, "configs", "controlled_duffing.json")
    stream = print

    use_saved_data = False

    kmt = KoopmanModelTrainer(config_path, curdir, save_model=True, save_sim_data=True, stream=stream)

    if use_saved_data:
        states, controls, t_vec = kmt.loadSimTrajectories(control=True)
        # pk_model = kmt.loadPyKoopmanModel()
    else:
        states, controls, t_vec = kmt.simTrajOpenLoop(genControlCmd, seed=4)

    if use_saved_data:
        pk_model = kmt.loadPyKoopmanModel()
    else:
        reg = regression.EDMDc()
        obs = observables.Polynomial(degree=5, include_bias=True)
        pk_model = kmt.trainPyKoopmanModel(states, controls=controls, regressor=reg, observables=obs)


    kAnalysis = KoopmanAnalysis(config_path, pk_model, stream=stream)
    kAnalysis.listEigVal()
    # _, valid_idx = kAnalysis.eigFunValidity(states, t_vec, alpha=1, score_threshold=2)
    

    # ====== LQR
    def dlqr(A, B, Q, R):
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        return K, P

    def sat(u, umax):
        return np.clip(u, -umax, umax)

    def koopman_lqr_u(x, K, pk_model):
        x1e = -2.0
        x_eq = np.array([x1e, 0.0])
        u_eq = x1e**3 - x1e   # feedforward that makes x_eq an equilibrium
        
        z = pk_model.observables.transform(x)
        z_ref = pk_model.observables.transform(x_eq.reshape(1, -1))
        v = -K @ (z - z_ref).T  # feedback term
        u = v + u_eq # total = feedforward + feedback
        # return sat(u, umax).reshape(1, -1)
        return u
    
    A = pk_model.A
    B = pk_model.B
    C = pk_model.C
    nz = A.shape[0]

    Qx = np.diag([10.0, 1.0])
    Q = C.T @ Qx @ C + 1e-6*np.eye(nz)
    R = np.array([[0.01]]) # tune this

    K, _ = dlqr(A, B, Q, R)
    # ======

    # kmt.simTrajClosedLoop(koopman_lqr_u, K=K, pk_model=pk_model)

    # return


    kAnalysis.timeReachBound(valid_ef_idx=np.array([19, 20, 9, 10, 14, 15]), n_samples=100)
    tested_traj, U = kAnalysis.verifyReachabilityWithSim_c(
        controller=koopman_lqr_u,
        n_samples=10, 
        final_time=4, 
        deltaT=0.01,
        pk_model=pk_model,
        K=K
    )


    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    a = 0.2
    rect1 = patches.Rectangle((1.68, 1.16), a, a, edgecolor='r', facecolor='none', zorder=3)
    rect2 = patches.Rectangle((-2.2, -0.3), 0.3, a, edgecolor='r', facecolor='none', zorder=3)
    ax[0].add_patch(rect1)
    ax[0].add_patch(rect2)

    for traj in tested_traj:
        ax[0].plot(traj[:, 0], traj[:, 1])

    ax[1].plot(np.arange(0, 4, 0.01), U[0].squeeze())

    plt.show()


if __name__ == "__main__":
    main()