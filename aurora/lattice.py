import numpy as np
class Lattice:
    def __init__(self, a1, a2, kind="square"):
        self.L = np.vstack((a1, a2)).T
        self.kind = kind
    
    def bz_path(self, points, sampling):
        key_points = {
            "hexa": {},
            "square": {"G": [0, 0], "X": [0.5, 0], "M": [0.5, 0.5]}
        }
        path = []
        keys = key_points[self.kind]
        current = keys[points[0]]
        for i in range(1, len(points)):
            target = keys[points[i]]
            res= sampling[i-1]
            x = np.linspace(current[0], target[0], res)
            y = np.linspace(current[1], target[1], res)
            for xx, yy in zip(x, y):
                path.append((xx, yy))
            current = keys[points[i]]
        
        return path