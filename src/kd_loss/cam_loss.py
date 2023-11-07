import torch
import torch.nn as nn
import numpy as np

# class CAMLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, student_cam, teacher_cam, batch_size, device):
#         cam_loss = 0
#         for i in range(batch_size):
            
#             s_cam = student_cam[i] / torch.norm(student_cam[i], p='fro')
#             t_cam = teacher_cam[i] / torch.norm(teacher_cam[i], p='fro')
#             loss = s_cam - t_cam
#             loss = torch.norm(loss, p='fro')
#             cam_loss += loss
#         # cam_loss = torch.tensor(cam_loss)
#         return cam_loss

class CAMLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, student_cam, teacher_cam, batch_size):
        loss_fn = nn.MSELoss()
        cam_loss = 0.
        
        for i in range(batch_size):
            s = torch.norm(student_cam[i], p='fro')
            t = torch.norm(teacher_cam[i], p='fro')
            
            s_cam = student_cam[i] / s
            t_cam = teacher_cam[i] / t
            
            loss = loss_fn(s_cam, t_cam)
            cam_loss += loss
        
        cam_loss /= batch_size
        return cam_loss