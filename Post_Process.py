from typing import final
import numpy as np
import cv2,sys,os
from PIL import Image
from scipy import sparse
from scipy.sparse import linalg
from eval import IoU


def post_processing(weight, is_multi, show=False):
    print('post processing start\nweight:{} is_multi:{}'.format(weight,is_multi))
    multi = 'multi' if is_multi else 'normal'
    if is_multi:
        scribble = cv2.imread('./datasets/Emily-In-Paris-scribbles-plus.png')
        gt = cv2.imread('./datasets/Emily-In-Paris-gt-plus.png')
    else:
        scribble = cv2.imread('./datasets/Emily-In-Paris-scribbles.png')
        gt = cv2.imread('./datasets/Emily-In-Paris-gt.png')
    scribble = scribble[:,:,2]
    shape = scribble.shape
    scribble = scribble.transpose().flatten()
    gt = gt[:,:,2]
    gt = gt.transpose().flatten()

    label_hat = np.zeros([8,shape[0]*shape[1]])

    for i in range(8):
        if is_multi:
            if i==0:continue 
        else:
            pass
        filename = './expected_label/expected_label{}_{}_{}.npy'.format(i,weight,multi)
        if os.path.exists(filename):
            l = np.load(filename)
            label_hat[i,:] = l
    
    label = np.zeros([8,shape[0]*shape[1]])

    if is_multi:

        for i in range(len(scribble)):
            pixel = scribble[i]
            if pixel == 0:
                continue
            label[pixel][i] = 1
    else:
        for i in range(len(scribble)):
            pixel = scribble[i]
            if pixel == 0:
                continue
            else:
                label[pixel-1][i] = 1

    

    
    final_label = np.zeros(len(scribble))
    
    for i in range(len(final_label)):
        ls = label[:,i]
        if not ls.any():
            final_label[i] = np.argmax(label_hat[1:,i])+1 if is_multi else np.argmax(label_hat[:,i])
            
        else:
            final_label[i] = np.argmax(ls)

    np.save('./output/final_label_{}_{}'.format(weight,multi),final_label)

    IoU(weight=weight,is_multi=is_multi,show=show)

