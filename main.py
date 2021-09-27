from scipy import sparse
from neighbor_mat import neighborhood_mat
import numpy as np
import cv2
import sys,os

'''
def neighborhood_mat(img):
    img = img.astype(np.int16)
    img = np.pad(img, (1,1),constant_values=-1)
    
    shape = img.shape
  
    result_mat = None
    
    for i in range(1, shape[0]-1):
        os.system('cls')
        print('{0}/{1}'.format(i,shape[0]-1))
        for j in range(1, shape[1]-1):
            r_s = img[i-1:i+2,j-1:j+2]
            
            num_padding = np.count_nonzero(r_s == -1)
            sum = np.sum(r_s) + num_padding - img[i][j]
            

            
            avg = round(sum / (8-num_padding),2)
            

            flattened = r_s.flatten()
            var = 0
            for k in range(len(flattened)):
                if flattened[k] == -1 or k==4:
                    continue
                else:
                    var += (flattened[k] - avg )**2
            var = var / (8-num_padding)
            var = round(var,2)


            n_mat = np.zeros((1,3,3))

            for n in range(3):
                for m in range(3):
                    w_rs = np.exp(-(r_s[1][1] - r_s[n][m])**2 / (2*var))
                    n_mat[0][n][m] = 0 if r_s[n][m]==-1 else w_rs
            
            
            if result_mat is None:
                result_mat = n_mat
            else:
                result_mat = np.append(result_mat, n_mat, axis=0)


    

           


        
        ##corners
        if i==0 and j==0:
            pass
        elif i==0 and j==shape[1]-1:
            pass
        elif i==shape[0]-1 and j==0:
            pass
        elif i==shape[0]-1 and j==shape[1]-1:
            pass

        ##edges
        if i==0 and j>0 and j<shape[1]:
            pass
        elif i>0 and i<shape[0] and j==0:
            pass
        elif i==shape[0]-1 and j>0 and j<shape[1]:
            pass
        elif i>0 and i<shape[0] and j==shape[1]-1:
            pass

        ##commons
        else:
            pass
        
    return result_mat

'''
def main():
    
    g_img = cv2.imread('./datasets/Emily-In-Paris-gray.png',cv2.IMREAD_GRAYSCALE,)
    n_mat = neighborhood_mat(g_img)
    print(n_mat.shape)
    sparse.save_npz('neighborhood_matrix.npz',n_mat)
    
    


    
    
    



    '''
    cv2.imshow('test',g_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''




if __name__ == "__main__":
    main()