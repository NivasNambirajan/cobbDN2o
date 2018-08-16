import pandas as pd
import numpy as np
from scipy.optimize import LinearConstraint, minimize, BFGS

class CalculateScalar(object):
    """
    This class calculates scalars for TP directly from data
    """
    def __init__(self, stack, a, b, is_b_tilda_positive):
        """
        Args:
            stack: dataframe
            a: float, coefficient for the first term in taylor series
            b: float, coefficient for the second term in taylor series
            is_b_tilda_positive: boolean, controls the sign of b_tilda
        """
        self.stack = stack
        self.a, self.b = a, b
        self.is_b_tilda_positive = is_b_tilda_positive

        self.spend_matrix, self.non_spend_matrix, self.outcome, self.high_modeled_units= self.split_spend_nonspend_outcome_var()
        self.scalar_calc_stack = self.stack_for_scalar_calc()
        self.scalar_lower_bound, self.scalar_upper_bound = self.calc_scalar_bounds()
        self.a_tilda_lower_bound, self.a_tilda_upper_bound = self.calc_a_tilda_bounds()
        self.constraints = self.linear_constraints()


    def split_spend_nonspend_outcome_var(self):
        """
        This function splits stack into TP, control, outcome and calculate TP high spend
        """
        columns = self.stack.columns
        spend_var = [var for var in columns if var.endswith('_SP')]
        non_spend_var = [var for var in columns if not var.endswith('_SP') and not var.startswith('O_')]
        outcome = [var for var in columns if var.startswith('O_')]
        spend_matrix = np.matrix(stack[spend_var])
        non_spend_matrix = np.matrix(stack[non_spend_var])
        outcome = stack[outcome]
        high_modeled_units = np.sum(spend_matrix, axis=0) / np.count_nonzero(spend_matrix, axis=0) + 2 * np.std(spend_matrix, axis=0)

        return spend_matrix, non_spend_matrix, outcome, high_modeled_units



    def stack_for_scalar_calc(self):
        """
        This function creates a stack based on talyor series equation. A bias term is also added
        """
        scalar_calc_stack = np.concatenate([self.spend_matrix, np.square(self.spend_matrix), self.non_spend_matrix, np.ones((self.spend_matrix.shape[0], 1))], axis=1)

        return scalar_calc_stack



    def calc_scalar_bounds(self):
        """
        This function calculates scalar bounds. scalar = high spend / (saturation - 1)
        """
        scalar_lower_bound = self.high_modeled_units / 9
        scalar_upper_bound = self.high_modeled_units

        return scalar_lower_bound, scalar_upper_bound



    def calc_a_tilda_bounds(self):
        """
        This function calculates scalar bounds. scalar = high spend / (saturation - 1)
        """
        if is_b_tilda_positive:
            a_tilda_lower_bound = (a / b) * self.high_modeled_units
            a_tilda_upper_bound = (a / 9.0 / b) * self.high_modeled_units

        else:
            a_tilda_lower_bound = (a / 9.0 / b) * self.high_modeled_units
            a_tilda_upper_bound = (a / b) * self.high_modeled_units

        return a_tilda_lower_bound, a_tilda_upper_bound



    def objective(self, beta):

        return sum((np.dot(np.asarray(self.scalar_calc_stack), beta) - np.squeeze(self.outcome)) ** 2)



    def linear_constraints(self):
        """
        This function creates constraints in a matrix format
        """
        num_spend_var = self.spend_matrix.shape[1]
        num_non_spend_var = self.non_spend_matrix.shape[1]

        identity_matrix = np.eye(num_spend_var)
        zero_matrix = np.zeros((num_spend_var, num_spend_var))

        lower_bound_matrix = np.concatenate([identity_matrix, -identity_matrix * np.array(self.a_tilda_lower_bound)[0]], axis=1)
        upper_bound_matrix = np.concatenate([identity_matrix, -identity_matrix * np.array(self.a_tilda_upper_bound)[0]], axis=1)

        a_tilda_constraint_matrix = np.concatenate([identity_matrix, zero_matrix], axis=1)
        b_tilda_constraint_matrix = np.concatenate([zero_matrix, identity_matrix], axis=1)

        constraint_matrix = np.concatenate([lower_bound_matrix, upper_bound_matrix, a_tilda_constraint_matrix, b_tilda_constraint_matrix], axis=0)
        constraint_matrix = np.concatenate([constraint_matrix, np.zeros((constraint_matrix.shape[0], num_non_spend_var + 1))], axis=1)

        if self.is_b_tilda_positive:
            lower_bound = [0] * num_spend_var + [-np.inf] * num_spend_var * 2 + [0] * num_spend_var
            upper_bound = [np.inf] * num_spend_var + [0] * num_spend_var * 2 + [np.inf] * num_spend_var
        else:
            lower_bound = [0] * num_spend_var + [-np.inf] * num_spend_var  + [0] * num_spend_var + [-np.inf] * num_spend_var
            upper_bound = [np.inf] * num_spend_var + [0] * num_spend_var + [np.inf] * num_spend_var + [0] * num_spend_var

        constraints = LinearConstraint(constraint_matrix, lower_bound, upper_bound)
        return constraints



    def get_scalar(self):
        """
        This function does constrained minimization and calculates scalars
        """
        num_spend_var = self.spend_matrix.shape[1]
        beta_init = np.ones(self.spend_matrix.shape[1] * 2 + self.non_spend_matrix.shape[1] + 1)
        res = minimize(self.objective, beta_init, method='trust-constr', jac='3-point', hess=BFGS(), constraints=self.constraints)
        a_tilda, b_tilda, control_beta = res.x[: num_spend_var], res.x[num_spend_var: num_spend_var * 2], res.x[num_spend_var * 2:]
        scalars = 1.0 / ((self.a / self.b) * (b_tilda / a_tilda))
        beta = a_tilda / a / scalars

        return scalars, beta



# a, b, is_b_tilda_positive = 1, -0.5, False
# stack = pd.read_csv(r"C:\Users\twu\Documents\Python_Scripts\MMM\test_stack.csv", index_col='X_DT')
# CalculateScalar = CalculateScalar(stack, a, b, is_b_tilda_positive)
# scalars, beta = CalculateScalar.get_scalar()
# print(scalars, beta)







