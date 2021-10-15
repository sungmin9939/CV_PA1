from math import isnan
from typing import final

from matplotlib.pyplot import flag
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
    #When running first time...
    g_img = cv2.imread('./datasets/Emily-In-Paris-gray.png',cv2.IMREAD_GRAYSCALE)

    #Calculate neighborhood matrix and save...
    weights = ['normal','Laplacian','ML_Laplacian','MLsum']
    multis = [True, False]
    for weight in weights:
        for multi in multis:
            print('weight function :{}\nis_multi :{}'.format(weight,multi))
            neighborhood_mat(g_img,weight=weight,is_multi=multi)
            least_square(weight,multi)
            post_processing(weight,multi)
            time.sleep(300)
    '''
    g_img = cv2.imread('./datasets/Emily-In-Paris-gray.png',cv2.IMREAD_GRAYSCALE)

    #Calculate neighborhood matrix and save...
    
    


    weights = ['MLsum']
    multis = [False]
    for weight in weights:
        for multi in multis:
            print('weight function :{}\nis_multi :{}'.format(weight,multi))
            neighborhood_mat(g_img,weight=weight,is_multi=multi)
            least_square(weight,multi)
            post_processing(weight,multi)
            time.sleep(300)

    


    
    




    

if __name__ == "__main__":
    main()