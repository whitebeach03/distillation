import cv2
import torch
import torch.nn as nn
import numpy as np
import tqdm

a = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], 
                  [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])

b = torch.tensor([[[10, 20, 13.0], [30, 0, 3], [10, 4, 1]],
                  [[10, 20, 13.0], [30, 0, 3], [10, 4, 1]]])

def cam_loss(a, b):
    norm_a = np.linalg.norm(a, ord=2)
    norm_b = np.linalg.norm(b, ord=2)
    
    s_cam = a / norm_a
    t_cam = b / norm_b
    
    loss = s_cam - t_cam
    loss_norm = np.linalg.norm(loss, ord=2)
    
    return loss_norm



