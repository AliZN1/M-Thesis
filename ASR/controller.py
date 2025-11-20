import numpy as np
import cvxpy as cp


class LinearMPC:
    def __init__(self, A, B, u_min, u_max, n_horizon=10):
        self.A = A
        self.B = B
        self.state_dim = A.shape[0]
        self.input_dim = B.shape[1]
        
        self.N = n_horizon
        self.Q = np.eye(self.state_dim)
        self.R = 0.1 * np.eye(self.input_dim)

        self.u_min = u_min
        self.u_max = u_max

    def run(self, x0, x_ref):
        if isinstance(x_ref, np.ndarray) and x_ref.ndim == 2:
            if x_ref.shape == (self.state_dim, 1):
                x_ref = x_ref * np.ones((self.state_dim, self.N))
            elif x_ref.shape[0] == self.state_dim:
                n = x_ref.shape[1]
                if n < self.N:
                    ref = np.zeros((self.state_dim, self.N))
                    ref[:, :n] = x_ref
                    ref[:, n:] = x_ref[:, n-1:n]
                    x_ref = ref
            else:
                raise ValueError("array x_ref expected to have shape of (state_dim, num_horizon_step)")
        else:
            raise ValueError("x_ref argument has wrong type! Expected type is np.ndarray.")

        x_var = cp.Variable((self.state_dim, self.N+1))
        u_var = cp.Variable((self.input_dim, self.N))

        cost = 0
        constraints = []
        constraints += [x_var[:, 0] == x0[:, 0]]

        for k in range(self.N):
            # Stage cost: x'Qx + u'Ru
            cost += cp.quad_form(x_var[:, k] - x_ref[:, k], self.Q) + cp.quad_form(u_var[:, k], self.R)
            # System dynamics
            constraints += [x_var[:, k+1] == self.A @ x_var[:, k] + self.B @ u_var[:, k]]
            # Input constraints
            constraints += [u_var[:, k] >= self.u_min, u_var[:, k] <= self.u_max]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: problem not solved to optimality, status = {problem.status}")
            # Fallback: zero input
            u_mpc = np.zeros((self.input_dim, 1))
        else:
            # First input of the optimal sequence (receding horizon)
            u_mpc = u_var[:, 0].value.reshape(self.input_dim, 1)

        return u_mpc


