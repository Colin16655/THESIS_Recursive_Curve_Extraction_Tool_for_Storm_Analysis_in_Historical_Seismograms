import numpy as np
import scipy.special

def state_transition_matrix(r):
    """
    Generate the state transition matrix F_r for a given order r.
    
    Args:
        r (int): Order of the model (e.g., r=2 for constant velocity, r=3 for constant acceleration, etc.)
    
    Returns:
        numpy.ndarray: The state transition matrix of shape (r, r).
    """
    # Compute coefficients using binomial coefficients from finite differences
    coefficients = [(-1)**(r-i) * scipy.special.comb(r, i) for i in range(r)]
    
    # Construct the transition matrix
    F = np.zeros((r, r))
    F[0, :] = coefficients  # First row is the finite difference equation
    
    # Fill the subdiagonal structure
    for i in range(1, r):
        F[i, i-1] = 1  # Shift state down one step
    
    return F

# Example usage
r = 4  # Example for order 4 (constant jerk model)
F_r = state_transition_matrix(r)
print(F_r)
