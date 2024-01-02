from tqdm import tqdm
import pickle
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from src.model import resnet_student, resnet_teacher, Student, Teacher
import torch.optim as optimizers
from sklearn.metrics import accuracy_score
from src.kd_loss.st import SoftTargetLoss
from src.kd_loss.cam_loss import CAMLoss
from src.utils import *

def main():
    
    teacher_student_loss = 0
    teacher_st_loss = 0
    teacher_cam05_loss = 0
    teacher_cam10_loss = 0
    
    iteration = 10
    
    for i in range(iteration):
        print(i)
        batch_size = 128
        seed_everything(100+i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data_dir   = './data/cifar10'
        transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset    = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        teacher = Teacher().to(device)
        student = Student().to(device)
        st      = Student().to(device)
        cam05   = Student().to(device)
        cam10   = Student().to(device)
        
        teacher.load_state_dict(torch.load('./logs/normal/teacher/200_' + str(i) + '.pth'))
        student.load_state_dict(torch.load('./logs/normal/student/200_' + str(i) + '.pth')) 
        st.load_state_dict(torch.load('./logs/normal/st/200_' + str(i) + '.pth'))
        cam05.load_state_dict(torch.load('./logs/normal/cam/05_200_' + str(i) + '.pth'))
        cam10.load_state_dict(torch.load('./logs/normal/cam/10_200_' + str(i) + '.pth'))
        
        loss_fn  = nn.CrossEntropyLoss()
        cam_loss = nn.MSELoss()
        
        teacher.eval()
        student.eval()
        st.eval()
        cam05.eval()
        cam10.eval()
       
        with torch.no_grad():           
            for images, labels in tqdm(dataloader, leave=False):
                images, labels  = images.to(device), labels.to(device)
                _, teacher_cams = teacher(images)
                _, student_cams = student(images)
                _, st_cams      = st(images)
                _, cam05_cams   = cam05(images)
                _, cam10_cams   = cam10(images)
                
                for j in range(batch_size):
                    image = images[j]
                    label = int(labels[j])
                    teacher_feature = teacher_cams[j].to(device)
                    student_feature = student_cams[j].to(device)
                    st_feature      = st_cams[j].to(device)
                    cam05_feature   = cam05_cams[j].to(device)
                    cam10_feature   = cam10_cams[j].to(device)
                    
                    teacher_cam = create_teacher_cam(image, label, teacher_feature, teacher)
                    student_cam = create_student_cam(image, label, student_feature, student)
                    st_cam      = create_student_cam(image, label, st_feature, st)
                    cam05_cam   = create_student_cam(image, label, cam05_feature, cam05)
                    cam10_cam   = create_student_cam(image, label, cam10_feature, cam10)
                    
                    # 1. 教師と生徒のLoss
                    teacher_student_loss += cam_loss(teacher_cam, student_cam)
                    # 2. 教師と従来法のLoss
                    teacher_st_loss      += cam_loss(teacher_cam, st_cam)
                    # 3. 教師と提案法(cam_rate=0.5)のLoss
                    teacher_cam05_loss   += cam_loss(teacher_cam, cam05_cam)
                    # 4. 教師と提案法(cam_rate=0.5->0)のLoss
                    teacher_cam10_loss   += cam_loss(teacher_cam, cam10_cam)
            print(teacher_student_loss)
           
    teacher_student_loss /= iteration
    teacher_st_loss      /= iteration
    teacher_cam05_loss   /= iteration
    teacher_cam10_loss   /= iteration
    
    print('Teacher & Student: '          + str(teacher_student_loss.numpy()))
    print('Teacher & Distillation: '     + str(teacher_st_loss.numpy()))
    print('Teacher & Proposed(0.5): '    + str(teacher_cam05_loss.numpy()))
    print('Teacher & Proposed(0.5->0): ' + str(teacher_cam10_loss.numpy()))

def create_teacher_cam(image, label, feature, model):
    attmap = np.array([])
    for i in range(10):
        weight = model.fc.weight[i]
        weight = weight.reshape(256, 1, 1)
        cam = feature * weight
        cam = cam.detach().cpu().numpy()
        cam = np.sum(cam, axis=0)
        cam = cv2.resize(cam, (32, 32))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        attmap = np.append(attmap, cam)
    attmap = torch.tensor(attmap)
    return attmap

def create_student_cam(image, label, feature, model):
    attmap = np.array([])
    for i in range(10):
        weight = model.fc.weight[i]
        weight = weight.reshape(64, 1, 1)
        cam = feature * weight
        cam = cam.detach().cpu().numpy()
        cam = np.sum(cam, axis=0)
        cam = cv2.resize(cam, (32, 32))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        attmap = np.append(attmap, cam)
    attmap = torch.tensor(attmap)
    return attmap


if __name__ == '__main__':
    main()