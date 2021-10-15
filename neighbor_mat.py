import numpy as np
from scipy import sparse
import os,sys,math,time
import cv2
from Others import sumML
def neighborhood_mat(img, weight=None, is_multi=False):
    '''
    weight parameter can be 'normal','Laplacian','ML_Laplacian','MLsum' 
    '''

    t_start = time.time()
    ##preprocess according to weight functions...
    if weight == 'Laplacian':
        mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        img = cv2.filter2D(img,-1,mask)
    if weight == 'ML_Laplacian' or weight == 'MLsum':
        mask1 = np.array([[1,-2,1]])
        mask2 = np.array([[1],[-2],[1]])
        lap1 = cv2.filter2D(img,-1,mask1)
        lap2 = cv2.filter2D(img,-1,mask2)
        img = lap1 ** 2 + lap2 ** 2
    
        

    ##applying padding to image for computatioanl convenience
    img = img.astype(np.int16)
    img = np.pad(img, (1,1),constant_values=-1)
    shape = img.shape

    if weight == 'MLsum':
        img = sumML(img)

    

    ##loading scribbles...
    if is_multi:
        scribble = cv2.imread('./datasets/Emily-In-Paris-scribbles-plus.png')
    else:
        scribble = cv2.imread('./datasets/Emily-In-Paris-scribbles.png')
    scribble = scribble[:,:,2]


    ##calculating neighborhood matrix...
    result_mat = None

    for m in range(1,shape[1]-1):
        print('{0}/{1}'.format(m,shape[1]-1))
        start = time.time()
        n_matseg_total = None
        for n in range(1,shape[0]-1):

            #where input scribble exists...
            if scribble[n-1][m-1] != 0:
                n_matseg = np.zeros([1, (shape[0]-2)*(shape[1]-2)])
                if n_matseg_total is None:
                    n_matseg_total = sparse.coo_matrix(n_matseg.transpose())
                else:
                    coo_n_matseg = sparse.coo_matrix(n_matseg.transpose())
                    n_matseg_total = sparse.hstack((n_matseg_total, coo_n_matseg))
                continue
            
            
            #where input scribble does not exists...
            r_s = img[n-1:n+2,m-1:m+2]
            r_s = r_s.flatten()
            idx = (shape[0]-2) * (m-1) + n
            n_matseg = np.zeros([1, (shape[0]-2)*(shape[1]-2)])
            var = 0
            if weight == 'normal':

                num_padding = np.count_nonzero(r_s == -1)
                sum = np.sum(r_s) + num_padding - r_s[4]
                avg = sum / (8-num_padding)
                

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
                    index = 0
                    share = int(i/3)
                    rest = i%3
                    index = idx+(shape[0]-2)*(rest-1) + (share-1)
                    weight_r_s = 0

                    if weight == 'Laplacian' or weight == 'ML_Laplacian' or weight == 'MLsum':
                        weight_r_s = r_s[i]
                    elif weight == 'normal':
                        weight_r_s = math.exp((-(r_s[i]-r_s[4])**2)/(2*var))
                        #weight_r_s = 1+(r_s[i]-avg)*(r_s[4]-avg)/var
                   
                    
                    
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
    t_end = time.time()
    print('total_time: {}'.format(t_end - t_start))
    multi = 'multi' if is_multi else 'normal'
    sparse.save_npz('./n_mat/neighborhood_matrix_{}_{}'.format(weight,multi),result_mat)

                    





            

