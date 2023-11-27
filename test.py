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
from src.model import resnet_student, resnet_teacher
import torch.optim as optimizers
from sklearn.metrics import accuracy_score
from src.kd_loss.st import SoftTargetLoss
from src.kd_loss.cam_loss import CAMLoss

def main():
    
    teacher_student_loss = 0
    teacher_st_loss = 0
    teacher_cam01_loss = 0
    teacher_cam02_loss = 0
    teacher_cam03_loss = 0
    
    iteration = 1
    
    for i in range(iteration):
        print(i)
        batch_size = 128
        np.random.seed(i)
        torch.manual_seed(i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data_dir = './data/cifar10'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        teacher = resnet_teacher().to(device)
        student = resnet_student().to(device)
        st = resnet_student().to(device)
        cam01 = resnet_student().to(device)
        cam02 = resnet_student().to(device)
        cam03 = resnet_student().to(device)
        
        teacher.load_state_dict(torch.load('./logs/resnet/teacher/' + str(i) + '.pth'))
        student.load_state_dict(torch.load('./logs/resnet/student/' + str(i) + '.pth'))
        st.load_state_dict(torch.load('./logs/resnet/st/' + str(i) + '.pth'))
        cam01.load_state_dict(torch.load('./logs/resnet/cam/01_' + str(i) + '.pth'))
        cam02.load_state_dict(torch.load('./logs/resnet/cam/02_' + str(i) + '.pth'))
        cam03.load_state_dict(torch.load('./logs/resnet/cam/03_' + str(i) + '.pth'))
        
        loss_fn = nn.CrossEntropyLoss()
        cam_loss = nn.MSELoss()
        
        teacher.eval()
        student.eval()
        st.eval()
        cam01.eval()
        cam02.eval()
       
        with torch.no_grad():           
            for images, labels in tqdm(dataloader, leave=False):
                images, labels = images.to(device), labels.to(device)
                _, teacher_cams = teacher(images)
                _, student_cams = student(images)
                _, st_cams = st(images)
                _, cam01_cams = cam01(images)
                _, cam02_cams = cam02(images)
                _, cam03_cams = cam03(images)
                
                for j in range(batch_size):
                    image = images[j]
                    label = int(labels[j])
                    teacher_feature = teacher_cams[j].to(device)
                    student_feature = student_cams[j].to(device)
                    st_feature = st_cams[j].to(device)
                    cam01_feature = cam01_cams[j].to(device)
                    cam02_feature = cam02_cams[j].to(device)
                    cam03_feature = cam03_cams[j].to(device)
                    
                    teacher_cam = create_teacher_cam(image, label, teacher_feature, teacher)
                    student_cam = create_student_cam(image, label, student_feature, student)
                    st_cam = create_student_cam(image, label, st_feature, st)
                    cam01_cam = create_student_cam(image, label, cam01_feature, cam01)
                    cam02_cam = create_student_cam(image, label, cam02_feature, cam02)
                    cam03_cam = create_student_cam(image, label, cam03_feature, cam02)
                    
                    # 1. 教師と生徒のLoss
                    teacher_student_loss += cam_loss(teacher_cam, student_cam)
                    # 2. 教師と従来法のLoss
                    teacher_st_loss      += cam_loss(teacher_cam, st_cam)
                    # 3. 教師と提案法(cam_rate=0.1)のLoss
                    teacher_cam01_loss   += cam_loss(teacher_cam, cam01_cam)
                    # 4. 教師と提案法(cam_rate=0.2)のLoss
                    teacher_cam02_loss   += cam_loss(teacher_cam, cam02_cam)
                    # 5. 教師と提案法(cam_rate=0.3)のLoss
                    # teacher_cam03_loss   += cam_loss(teacher_cam, cam03_cam)
            
    teacher_student_loss /= iteration
    teacher_st_loss /= iteration
    teacher_cam01_loss /= iteration
    teacher_cam02_loss /= iteration
    teacher_cam03_loss /= iteration
    
    print('Teacher & Student: '            + str(teacher_student_loss.numpy()))
    print('Teacher & Distillation: '       + str(teacher_st_loss.numpy()))
    print('Teacher & Proposed(rate=0.1): ' + str(teacher_cam01_loss.numpy()))
    print('Teacher & Proposed(rate=0.2): ' + str(teacher_cam02_loss.numpy()))
    print('Teacher & Proposed(rate=0.3): ' + str(teacher_cam03_loss.numpy()))

def create_teacher_cam(image, label, feature, model):
    weight = model.fc.weight[label]
    weight = weight.reshape(1024, 1, 1)
    cam = feature * weight  
    cam = cam.detach().cpu().numpy()
    cam = np.sum(cam, axis=0)
    cam = cv2.resize(cam, (32, 32))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = torch.tensor(cam)
    return cam

def create_student_cam(image, label, feature, model):
    weight = model.fc.weight[label]
    weight = weight.reshape(512, 1, 1)
    cam = feature * weight  
    cam = cam.detach().cpu().numpy()
    cam = np.sum(cam, axis=0)
    cam = cv2.resize(cam, (32, 32))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = torch.tensor(cam)
    return cam


if __name__ == '__main__':
    main()