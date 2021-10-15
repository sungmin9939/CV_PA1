
from typing import final
import numpy as np
import cv2,sys,os
from PIL import Image
from scipy import sparse
from scipy.sparse import linalg


def IoU(weight, is_multi, show=False):
    
    multi = 'multi' if is_multi else 'normal'
    final_label = np.load('./output/final_label_{}_{}.npy'.format(weight,multi))
    if is_multi:
        gt = cv2.imread('./datasets/Emily-In-Paris-gt-plus.png')
    else:
        gt = cv2.imread('./datasets/Emily-In-Paris-gt.png')
    
    shape = gt.shape
    gt = gt[:,:,2]
    gt = gt.transpose().flatten()

    if is_multi:
        union = np.zeros(8)
        inter = np.zeros(8)
        for i in range(8):
            if i==0:continue

            for j in range(len(final_label)):
                l = final_label[j]
                gl = gt[j]

                if l == gl == 0:
                    continue
                elif l == gl == i:
                    union[i] += 1
                    inter[i] += 1
                    continue
                elif l == i or gl == i:
                    union[i] += 1
        iou = inter/union
        print(iou[1:])
        iou = np.sum(iou[1:])/(len(iou)-1)
        print(
            'IoU of {} labels using {} method is {}%'.format(multi, weight, round(iou*100,2))
            )
        print()
    else:
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
        print('IoU of {} labels using {} method is {}%'.format(multi, weight, round((inter/union)*100,2)))
        

    #image showing of multi label is not implemented
    if show:
        final_label = np.reshape(final_label, [shape[1],shape[0]]).transpose()
        final_label = final_label.astype(np.uint8)
        for i in range(final_label.shape[0]):
            for j in range(final_label.shape[1]):
                if final_label[i][j] == 1:
                    final_label[i][j] = 255
    
        img = Image.fromarray(final_label)
        img.save('./results/result_{}_{}.png'.format(weight,multi),'PNG')
        img.show()

    
