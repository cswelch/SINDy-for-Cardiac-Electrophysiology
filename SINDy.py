import numpy as np
from scipy.integrate import odeint
from itertools import combinations_with_replacement
from math import factorial

class SINDy():
    """SINDy class to determine active terms in function space of a 
    given dataset of a dynamic system.
    """
    
    def __init__(self, lambda_=0.025, n=10, poly_power=3, feature_names='xyz'):
        self.lambda_ = lambda_
        self.n = n
        self.poly_power = poly_power
        self._m = None
        self._Theta = None
        self._Xi = None
        self._functions = []
        self._feature_names = feature_names
        
    def fit(self, X, dx_dt):
        self._Xi = self.sparsify_dynamics(X, dx_dt)
        self._functions = self.function_vector()

        print("Functions:\n", self._functions)
        print("\nFit coefficients (Xi):\n", self._Xi)

        Xi = np.column_stack((self._functions, self._Xi.astype(dtype=np.dtype("<U6"))))
        return Xi
        
    def polynomial_combination(self, states, degrees):
        combinations = factorial(degrees + states) / (factorial(degrees) * factorial(states))
        return int(combinations)
    
    def Theta_library(self, X):
        self._states = X.shape[1]
        self._m = X.shape[0]
        poly_space = self.polynomial_combination(self._states, degrees=self.poly_power)
        self._Theta = np.ones((self._m, poly_space))
        counter = 1
        column_idx = list(range(self._states))
        for i in range(self.poly_power+1):
            for combo in combinations_with_replacement(column_idx, i):
                temp_ones= np.ones((self._m, 1))
                for c in combo:
                    temp_ones = temp_ones * X[:, c].reshape(self._m, 1)
                self._Theta[:, (counter - 1):counter] = temp_ones
                counter += 1
        return self._Theta
    
    def sparsify_dynamics(self, X, dx_dt,):
        self._Theta = self.Theta_library(X)
        Xi = np.linalg.lstsq(self._Theta, dx_dt, rcond=None)[0]

        print(dx_dt.shape)
        print(self._Theta.shape)
        print(Xi.shape)
        
        for i in range(self.n):
            Xi = np.where(np.abs(Xi) < self.lambda_, 0, Xi)
            for j in range(Xi.shape[1]):
                mask = np.abs(Xi[:, j]) > self.lambda_
                Xi[:, j][mask] = np.linalg.lstsq(self._Theta[:, mask], dx_dt[:, j], rcond=None)[0]
        return Xi
    
    def function_vector(self):
        for i in range(self.poly_power+1):
            if i == 0:
                self._functions.append("1")
            else:
                for combo in combinations_with_replacement(self._feature_names, i):
                    self._functions.append("".join(combo))
        return np.array(self._functions)
    
    def equations(self):
        eqn_system = []
        feature_names_list = list(self._feature_names)     # Convert single string into list containing each constituent variable/character
        for i in range(self._states):
            eqn = self._functions[np.argwhere(np.abs(self._Xi[:, i]) > 0)].flatten()
            coeff = self._Xi[:, i][np.argwhere(np.abs(self._Xi[:, i]) > 0)].flatten()
            coeff = coeff.astype(dtype="<U6")
            p=feature_names_list[i] + "\' = "
            for j in range(len(eqn)):
                if j != list(range(len(eqn)))[-1]:
                    p = p + str(coeff[j] + eqn[j] + " + ")
                else:
                    p = p + str(coeff[j] + eqn[j])
            eqn_system.append(p)
            
        return eqn_system