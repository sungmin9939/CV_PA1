from scipy import sparse
import numpy as np
import cv2, time,sys
import scipy.sparse.linalg as linalg

def least_square():
    neighbor = sparse.load_npz('neighborhood_matrix_opt1.npz')
    neighbor = neighbor.transpose()
    Id = sparse.identity(neighbor.shape[0])
    Id = sparse.coo_matrix(Id - neighbor)

    scribble = cv2.imread('./datasets/Emily-In-Paris-scribbles.png')
    
    scribble = scribble[:,:,2]

    shape = scribble.shape
    l1 = np.zeros(shape[0]*shape[1])
    l2 = np.zeros(shape[0]*shape[1])

    scribble = scribble.transpose().flatten()
    for i in range(len(scribble)):
        pixel = scribble[i]
        if pixel == 0:
            continue
        if pixel == 1:
            l1[i] = 1
        elif pixel == 2:
            l2[i] = 1

   
 
    print('calculation start...')
    start = time.time()

    
    expected_label1 = linalg.lsqr(Id,l1)
    np.save('./expected_label1',expected_label1[0])

    print(expected_label1[0])
    print('time: {}'.format(time.time() - start))
    

    expected_label2 = linalg.lsqr(Id,l2)
    
    np.save('./expected_label2',expected_label2[0])
    print('------------------------------------')
    print(expected_label2[0])
    print('time: {}'.format(time.time() - start))
    
    