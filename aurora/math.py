from numpy import array
from math import cos, sin

def rotation_matrix(theta):
    ''' Obtenir une matrice de rotation pour n'importe quel angle en radians '''
    return array([[cos(theta), - sin(theta)], 
                  [sin(theta),   cos(theta)]
                 ])
