import imcut.pycut
import numpy as np
import cv2
import sys
from PIL import Image
img = cv2.imread('./datasets/Emily-In-Paris-gray.png',cv2.IMREAD_GRAYSCALE)
scribble = cv2.imread('./datasets/Emily-In-Paris-scribbles.png')
scribble = scribble[:,:,2]
gt = cv2.imread('./datasets/Emily-In-Paris-gt.png')
gt = gt[:,:,2]



img = np.expand_dims(img, axis=-1)
scribble = np.expand_dims(scribble, axis=-1)
gc = imcut.pycut.ImageGraphCut(img)
gc.set_seeds(scribble)
gc.run()
result = gc.segmentation.squeeze()

result_img = np.zeros(result.shape)
inter = 0
union = 0
print(result_img.shape)
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if result[i][j] == gt[i][j] == 0:
            continue
        elif result[i][j] == gt[i][j] == 1:
            union += 1
            inter += 1
            result_img[i][j] = 255
        else:
            union += 1
            result_img[i][j] = 255 if result[i][j] else 0

print("graph cut IoU: {}".format(inter/union))
result_img = result_img.astype(np.int16)
result_img = Image.fromarray(result_img)
result_img.save('./results/result_graphcut_normal.png','PNG')
result_img.show()
