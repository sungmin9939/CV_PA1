import imcut.pycut
import numpy as np

im = np.random.random([5, 5, 1])
im[:3, :3] += 1.

seeds = np.zeros([5, 5, 1], dtype=np.uint8)
seeds[:3,0] = 1  # foreground
seeds[:3,4] = 2  # background

gc = imcut.pycut.ImageGraphCut(im)
gc.set_seeds(seeds)
gc.run()

print(gc.segmentation.squeeze())