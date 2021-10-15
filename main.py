from math import isnan
from typing import final
import time
from matplotlib.pyplot import flag
from neighbor_mat import neighborhood_mat
import numpy as np
import cv2
from scipy import sparse
from Least_Square import least_square
from Post_Process import post_processing

import scipy.sparse.linalg as linalg


def main():
    '''
    All results are already saved. If you want to re-run whole stuff, follow the codes below
    '''


    #When running first time...
    g_img = cv2.imread('./datasets/Emily-In-Paris-gray.png',cv2.IMREAD_GRAYSCALE)

    #Calculate neighborhood matrix and save...
    weights = ['normal','Laplacian','ML_Laplacian','MLsum']
    
    for weight in weights:
        
        print('weight function :{}\nis_multi :{}'.format(weight,'FALSE'))
        neighborhood_mat(g_img,weight=weight,is_multi=False)
        least_square(weight,False)
        post_processing(weight,False)
        time.sleep(300)
    for weight in weights[:2]:
        neighborhood_mat(g_img,weight=weight,is_multi=True)
        least_square(weight,True)
        post_processing(weight,True)

if __name__ == "__main__":
    main()