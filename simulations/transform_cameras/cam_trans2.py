# -*- coding: utf-8 -*-
"""
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle 
import os

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    
def rotation(x, y, z):
    """Rotate point p by angle x, y, z"""
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                    [0, np.sin(x), np.cos(x)]])
                    
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1 , 0],
                    [-np.sin(y), 0 , np.cos(y)]])       
                    
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z) , 0],
                    [0, 0, 1]])
                    
    
                    
    R = np.dot(Rx, np.dot(Ry, Rz))
   
    return R

pars = pickle.load(open('F:\\Postural Control Project\\calibration\\stereo_camera_calib_params.pkl', 'rb'), encoding = "Latin-1")

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T_, E, F, P1, P2 = pars


T = T_.flatten()
#R2 = rotation(0, 0, 0)
#
#angle_between(np.array([0,0,1]), T)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(0,0,0)
ax.plot([0, 0], [0, 0], [0, 1500])

ax.scatter(T[0],T[1], T[2])
ax.plot([0, T[0]], [0, T[1]], [0, T[2]])

new_point = np.dot(R, np.array([0,0,1500]) ) + T
ax.plot([T[0], new_point[0]], [T[1], new_point[1]], [T[2], new_point[2]] )
##ax.plot([0, new_point[0]], [0, new_point[1]], [0, new_point[2]] )
T2 = T/2.0
ax.scatter(T2[0],T2[1], T2[2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim([0, 1500])
ax.set_ylim([0, 1500])
ax.set_zlim([0, 1500])