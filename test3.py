import numpy as np
import cv2
from PIL import Image
final_label = np.load('./output/final_label_ML_Laplacian_normal.npy')

print(np.count_nonzero(final_label == 1))
gt = cv2.imread('./datasets/Emily-In-Paris-gt.png')
gt = gt[:,:,2]
shape = gt.shape
gt = gt.transpose().flatten()

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



final_label = np.reshape(final_label, [shape[1],shape[0]]).transpose()
final_label = final_label.astype(np.uint8)
for i in range(final_label.shape[0]):
    for j in range(final_label.shape[1]):
        if final_label[i][j] == 1:
            final_label[i][j] = 255

img = Image.fromarray(final_label)
img.save('result.png','PNG')
img.show()