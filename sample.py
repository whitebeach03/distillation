import cv2
import torch
import torch.nn as nn
import numpy as np
import tqdm

student_cam = np.array([])
teacher_cam = np.array([])

a = np.array([[1, 2, 3],
              [1, 2, 3]])

b = np.array([[10, 20, 30],
              [10, 20, 30]])

student_cam = np.append(student_cam, a)
student_cam = np.append(student_cam, b)
print(student_cam)

c = np.array([[100, 200, 300],
              [100, 200, 300]])

d = np.array([[1000, 2000, 3000],
              [1000, 2000, 3000]])

teacher_cam = np.append(teacher_cam, c)
teacher_cam = np.append(teacher_cam, d)
print(teacher_cam)

loss_fn = nn.MSELoss()
student_cam = torch.Tensor(student_cam)
teacher_cam = torch.Tensor(teacher_cam)

loss = loss_fn(student_cam, teacher_cam)
print(loss)