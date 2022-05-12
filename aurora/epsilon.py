from PIL import ImageDraw, Image
import numpy as np
from math import cos, sin, pi, sqrt
from numpy.linalg import norm, inv

from aurora.math import rotation_matrix

class Layer:
    def __init__(self, a1, a2, resolution, supersample=1, size=(1, 1)) -> None:
        self.L = np.vstack((a1, a2)).T
        self.resolution = resolution
        self.size = np.asarray(size)
        self.draw_resolution = tuple(e * supersample for e in resolution)
        self._epsilon = Image.fromarray(np.zeros(self.draw_resolution))
        self.drawing = ImageDraw.Draw(self._epsilon)
    
    def block(self, center, size, e1=[1, 0], e2=[0, 1]):
        ''' Converts rectangle center position and size to the polygon points in lattice space '''
        w, l = np.array(size)
        e1 = self.L @ np.array(e1)
        e2 = self.L @ np.array(e2)
        e1 /= np.linalg.norm(e1)
        e2 /= np.linalg.norm(e2)
        #R = rotation_matrix(theta)
        xy = self.L @ np.asarray(center) - w / 2.0 * e1 - l / 2.0 * e2
        xy1 = xy + w * e1
        xy2 = xy + l * e2
        xy3 = xy + w * e1 + l * e2
        LI = np.linalg.inv(self.L)
        return [ LI @ e / self.size for e in [ xy, xy2, xy3, xy1 ] ]


    def draw_block(self, center, size, e1=[1, 0], e2=[0, 1], theta=0):
        R = rotation_matrix(theta)
        vertices = self.block(center, size, R @ e1, R @ e2)
        self.drawing.polygon([ (v[0]*self.draw_resolution[0], v[1]*self.draw_resolution[1]) for v in vertices], fill=True)
    @property
    def epsilon(self):
        output = np.array(self._epsilon.resize(self.resolution))
        output /= np.max(output)
        #output[output > 0.95] = 0.95
        #output /= np.max(output)
        output[output<0.0] = 0.0
        return output
    
    @property
    def fourier(self):
        return np.fft.fftshift(np.fft.fft2(self.epsilon / np.prod(self.epsilon.shape)))
