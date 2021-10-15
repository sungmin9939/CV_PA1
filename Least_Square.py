from scipy import sparse
import numpy as np
import cv2, time,sys
import scipy.sparse.linalg as linalg

def least_square(weight, is_multi):
    multi = 'multi' if is_multi else 'normal'
    neighbor = sparse.load_npz('./n_mat/neighborhood_matrix_{}_{}.npz'.format(weight,multi))
    neighbor = neighbor.transpose()
    Id = sparse.identity(neighbor.shape[0])
    Id = sparse.coo_matrix(Id - neighbor)

    if is_multi:
        scribble = cv2.imread('./datasets/Emily-In-Paris-scribbles-plus.png')
    else:
        scribble = cv2.imread('./datasets/Emily-In-Paris-scribbles.png')
    
    scribble = scribble[:,:,2]

    shape = scribble.shape
    labels = np.zeros([8, shape[0]*shape[1]])

    

    scribble = scribble.transpose().flatten()
    
    if is_multi:

        for i in range(len(scribble)):
            pixel = scribble[i]
            if pixel == 0:
                continue
            labels[pixel][i] = 1
    else:
        for i in range(len(scribble)):
            pixel = scribble[i]
            if pixel == 0:
                continue
            else:
                labels[pixel-1][i] = 1

   

   
 
    print('calculation start...')
    start = time.time()

    for i in range(8):
        if i>1 and not is_multi:
            continue
        if i==0 and is_multi:
            continue
        expected_label = linalg.lsqr(Id, labels[i,:])
        np.save('./expected_label/expected_label{}_{}_{}'.format(i, weight,multi),expected_label[0])
        print('{}th label finished. time:{}'.format(i, time.time()-start))
        time.sleep(60)
    
    
    
    