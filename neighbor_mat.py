import numpy as np
from scipy import sparse
import os,sys,math,time


def neighborhood_mat(img):
    img = img.astype(np.int16)
    img = np.pad(img, (1,1),constant_values=-1)
    
    shape = img.shape
  
    result_mat = None

    for m in range(1,shape[1]-1):
        #os.system('cls')
        print('{0}/{1}'.format(m,shape[1]-1))
        start = time.time()
        for n in range(1,shape[0]-1):
            
            r_s = img[n-1:n+2,m-1:m+2]
            r_s = r_s.transpose().flatten()
            idx = (shape[0]-2) * (m-1) + n
            num_padding = np.count_nonzero(r_s == -1)
            
            sum = np.sum(r_s) + num_padding - r_s[4]
            avg = sum / (8-num_padding)
            
            var = 0

            for i in range(len(r_s)):
                if r_s[i] == -1 or i==4:
                    continue
                else:
                    var += (r_s[i] - avg)**2
            var = (var/(8-num_padding))

            n_matseg = np.zeros([1, (shape[0]-2)*(shape[1]-2)])

            for i in range(len(r_s)):
                if r_s[i] == -1 or i==4:
                    continue
                else:
                    weight_r_s = math.exp(-(r_s[i]-r_s[4])**2/(2*var))
                    index = 0
                    indi = i/3

                    if indi == 0:
                        index = idx-(shape[0]-2) + (i%3) - 1
                    elif indi == 1:
                        index = idx + (i%3) - 1
                    elif indi == 2:
                        index = idx + (shape[0]-2) + (i%3) - 1
                    n_matseg[0][index] = weight_r_s
            if result_mat is None:
                result_mat = sparse.coo_matrix(n_matseg.transpose())
            else:
                coo_n_matseg = sparse.coo_matrix(n_matseg.transpose())
                result_mat = sparse.hstack((result_mat,coo_n_matseg))
        end = time.time()
        print('time: {}'.format(end-start))
    return result_mat

                    





            

