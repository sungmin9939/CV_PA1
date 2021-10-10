from typing import final
import numpy as np
import cv2,sys
from PIL import Image
from scipy import sparse
from scipy.sparse import linalg


def post_processing(show=False):


    l1_hat = np.load('expected_label1.npy')
    l2_hat = np.load('expected_label2.npy')

    scribble = cv2.imread('./datasets/Emily-In-Paris-scribbles.png')
    scribble = scribble[:,:,2]
    shape = scribble.shape
    scribble = scribble.transpose().flatten()

    gt = cv2.imread('./datasets/Emily-In-Paris-gt.png')
    gt = gt[:,:,2]
    gt = gt.transpose().flatten()

    
    l1 = np.zeros(shape[0]*shape[1])
    l2 = np.zeros(shape[0]*shape[1])

    for i in range(len(scribble)):
        pixel = scribble[i]
        if pixel == 0:
            continue
        if pixel == 1:
            l1[i] = 1
        elif pixel == 2:
            l2[i] = 1

    l1_hat = np.reshape(l1_hat,[1,-1])
    l2_hat = np.reshape(l2_hat,[1,-1])
    l12_hat = np.append(l1_hat,l2_hat,axis=0)

    l1 = np.reshape(l1, [1,-1])
    l2 = np.reshape(l2, [1,-1])
    l12 = np.append(l1,l2,axis=0)


    final_label = np.zeros(shape[0]*shape[1])
    for i in range(shape[0]*shape[1]):
        l1 = l12[0][i]
        l2 = l12[1][i]

        if l1 == l2 == 0:
            #print('{}th pixel => l1: {}, l2: {}. gt is {}'.format(i, l12_hat[0][i],l12_hat[1][i],gt[i]))

            arg = np.argmax(l12_hat[:,i])
            final_label[i] = arg
        else:
            final_label[i] = l2 if l2 == 1 else 0


    np.save('./final_label',final_label)

    union = 0
    inter = 0

    for i in range(len(final_label)):
        l_hat = final_label[i]
        gt_l = gt[i]

        if l_hat == gt_l == 0:
            continue
        elif l_hat == gt_l == 1:
            union += 1
            inter += 1
        else:
            union += 1
    print('IoU: {}'.format(inter/union * 100))


    if show:
        final_label = np.reshape(final_label, [shape[1],shape[0]]).transpose()
        final_label = final_label.astype(np.uint8)
        for i in range(final_label.shape[0]):
            for j in range(final_label.shape[1]):
                if final_label[i][j] == 1:
                    final_label[i][j] = 255
    
        img = Image.fromarray(final_label)
        img.save('result.png','PNG')
        img.show()

