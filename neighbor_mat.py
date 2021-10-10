import numpy as np
from scipy import sparse
import os,sys,math,time


def neighborhood_mat(img):
    t_start = time.time()
    img = img.astype(np.int16)
    img = np.pad(img, (1,1),constant_values=-1)
    
    shape = img.shape
  
    result_mat = None

    for m in range(1,shape[1]-1):
        #os.system('cls')
        print('{0}/{1}'.format(m,shape[1]-1))
        start = time.time()
        n_matseg_total = None
        for n in range(1,shape[0]-1):
            
            r_s = img[n-1:n+2,m-1:m+2]
            r_s = r_s.flatten()
            idx = (shape[0]-2) * (m-1) + n
            num_padding = np.count_nonzero(r_s == -1)
            n_matseg = np.zeros([1, (shape[0]-2)*(shape[1]-2)])
            
            sum = np.sum(r_s) + num_padding - r_s[4]
            avg = sum / (8-num_padding)
            var = 0

            for i in range(len(r_s)):
                if r_s[i] == -1 or i==4:
                    continue
                else:
                    var += (r_s[i] - avg)**2
            var = (var/(8-num_padding))
            if var == 0:
                var = 1e-9

            
            

            for i in range(len(r_s)):
                if r_s[i] == -1 or i==4:
                    continue
                else:
                    weight_r_s = math.exp((-(r_s[i]-r_s[4])**2)/(2*var))
                    #weight_r_s = 1+(r_s[i]-avg)*(r_s[4]-avg)/var
                   
                    if math.isnan(weight_r_s):
                        print('nan detected')
                        print(var)
                    index = 0
                    share = int(i/3)
                    rest = i%3

                    index = idx+(shape[0]-2)*(rest-1) + (share-1)
                    
                    n_matseg[0][index-1] = weight_r_s
            if np.sum(n_matseg) != 0:
                n_matseg = n_matseg/np.sum(n_matseg)
            
            if n_matseg_total is None:
                n_matseg_total = sparse.coo_matrix(n_matseg.transpose())
            else:
                coo_n_matseg = sparse.coo_matrix(n_matseg.transpose())
                n_matseg_total = sparse.hstack((n_matseg_total, coo_n_matseg))
        if result_mat is None:
            result_mat = n_matseg_total
        else:
            result_mat = sparse.hstack((result_mat, n_matseg_total))
        end = time.time()
        print('time: {}'.format(end-start))
    t_end = time.time()
    print('total_time: {}'.format(t_end - t_start))
    return result_mat

                    





            

