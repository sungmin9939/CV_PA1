from math import isnan
from typing import final
from neighbor_mat import neighborhood_mat
import numpy as np
import cv2
import sys,os,time
from scipy import sparse
from Least_Square import least_square
from Post_Process import post_processing

import scipy.sparse.linalg as linalg


def main():
    '''
    When running first time...
    g_img = cv2.imread('./datasets/Emily-In-Paris-gray.png',cv2.IMREAD_GRAYSCALE)

    #Calculate neighborhood matrix and save...
    n_mat = neighborhood_mat(g_img) 
    sparse.save_npz('neighborhood_matrix.npz',n_mat)
[]
    #Calculate least square solution...
    least_square()

    #Make final label and Calculate IoU
    '''
    post_processing(show=True)




    

if __name__ == "__main__":
    main()