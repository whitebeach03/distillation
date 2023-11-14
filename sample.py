import cv2
import torch
import torch.nn as nn
import numpy as np
import tqdm

a = torch.randn(3, 2, 2)
b = torch.randn(3, 1, 1)
print(a)
print(b)
c = a * b
print(c)