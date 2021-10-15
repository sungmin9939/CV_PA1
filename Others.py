import numpy as np

def sumML(img):
    shape = img.shape
    result = np.zeros(shape)
    result -= 1
    for n in range(1,shape[0]-1):
        for m in range(1,shape[1]-1):
            r_s = img[n-1:n+2,m-1:m+2]
            sum_r_s = np.sum(r_s) + np.count_nonzero(r_s == -1) - r_s[1][1]
            result[n][m] = sum_r_s

    return result
