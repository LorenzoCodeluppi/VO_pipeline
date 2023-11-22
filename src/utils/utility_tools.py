import numpy as np


def cross2Matrix(x):
  """ Antisymmetric matrix corresponding to a 3-vector
    Computes the antisymmetric matrix M corresponding to a 3-vector x such
    that M*y = cross(x,y) for all 3-vectors y.

    Input: 
      - x np.ndarray(3,1) : vector

    Output: 
      - M np.ndarray(3,3) : antisymmetric matrix
  """
  M = np.array([[0,   -x[2], x[1]], 
                [x[2],  0,  -x[0]],
                [-x[1], x[0],  0]])
  return M
