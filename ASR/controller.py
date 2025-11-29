import numpy as np
import cvxpy as cp


def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

class KoopmanMPC:
    def __init__(self, pk_model, u_min, u_max, n_horizon=10):
        self.lift = pk_model.observables.transform
        self.A = pk_model.A # (nz, nz)
        self.B = pk_model.B # (nz, nu)
        self.C = pk_model.C # (nx, nz)

        self.nz = self.A.shape[0]
        self.nx = self.C.shape[0]
        self.nu = self.B.shape[1]
        
        self.Np = n_horizon
        self.Q = np.diag([1.0, 1.0, 1.0])
        self.R = np.diag([0.01, 0.25])

        self.u_min = u_min
        self.u_max = u_max

    def run(self, x0, x_ref=None):
        if x_ref is None:
            x_ref = np.zeros((self.nx, self.Np))

        # =======
        x_ref_wrapped = x_ref.copy()
        x_ref_wrapped[2, :] = wrap_angle(x_ref[2, :])
        # =======

        z0 = self.lift(x0).squeeze()

        z_var = cp.Variable((self.nz, self.Np+1))
        u_var = cp.Variable((self.nu, self.Np))

        constraints = [z_var[:, 0] == z0]

        cost = 0
        for k in range(self.Np):
            # System dynamics
            constraints += [z_var[:, k+1] == (self.A @ z_var[:, k] + self.B @ u_var[:, k])]

            # State in original coordinates
            x_k = self.C @ z_var[:, k]

            # Input bounds: v in [v_min, v_max], gamma in [-gamma_max, gamma_max]
            constraints += [u_var[:, k] >= self.u_min, u_var[:, k] <= self.u_max]

            # Stage cost: (x_k - x_ref_k)' Q (x_k - x_ref_k) + u'Ru
            cost += cp.quad_form(x_k - x_ref[:, k], self.Q) + cp.quad_form(u_var[:, k], self.R)

            #terminal const on x_Np
            # x_N = self.C @ z_var[:, self.Np]
            # cost += cp.quad_form(x_N - x_ref[:, -1], self.Q/100)


        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: problem not solved to optimality, status = {problem.status}")
            # Fallback: zero input
            u_mpc = np.zeros((self.nu, 1))
        else:
            # First input of the optimal sequence (receding horizon)
            u_mpc = u_var[:, 0].value.reshape(self.nu, 1)

        return u_mpc


