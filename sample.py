import cv2
import torch
import torch.nn as nn
import numpy as np
import tqdm

cams = np.array([])

a = np.array([1, 2, 3],)

for i in range(3):
    cams = np.append(cams, a)

print(cams.shape)