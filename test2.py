import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('./datasets/Emily-In-Paris-scribbles-plus.png')

img = img[:,:,2]
print(np.count_nonzero(img==0))