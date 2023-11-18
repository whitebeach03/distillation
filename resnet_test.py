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
    teacher_testmodel_loss = 0
    teacher_st_loss = 0
    teacher_sample_loss = 0
    teacher_proposed_loss = 0
    
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
        test_model = resnet_student().to(device)
        st = resnet_student().to(device)
        # sample = resnet_student().to(device)
        proposed = resnet_student().to(device)
        
        teacher.load_state_dict(torch.load('./logs/resnet/teacher/' + str(i) + '.pth'))
        student.load_state_dict(torch.load('./logs/resnet/student/' + str(i) + '.pth'))
        test_model.load_state_dict(torch.load('./logs/resnet/cam/' + str(i) + '.pth'))
        st.load_state_dict(torch.load('./logs/resnet/st/' + str(i) + '.pth'))
        # sample.load_state_dict(torch.load('./logs/resnet/sample/' + str(i) + '.pth'))
        proposed.load_state_dict(torch.load('./logs/resnet/cam/10.pth'))
        
        loss_fn = nn.CrossEntropyLoss()
        cam_loss = nn.MSELoss()
        
        teacher.eval()
        student.eval()
        test_model.eval()
        st.eval()
        # sample.eval()
        proposed.eval()
        
        with torch.no_grad():           
            for images, labels in tqdm(dataloader, leave=False):
                images, labels = images.to(device), labels.to(device)
                _, teacher_cams = teacher(images)
                _, student_cams = student(images)
                _, testmodel_cams = test_model(images)
                _, st_cams = st(images)
                # _, sample_cams = sample(images)
                _, proposed_cams = proposed(images)
                
                for j in range(batch_size):
                    image = images[j]
                    label = int(labels[j])
                    teacher_feature = teacher_cams[i].to(device)
                    student_feature = student_cams[i].to(device)
                    testmodel_feature = testmodel_cams[i].to(device)
                    st_feature = st_cams[i].to(device)
                    # sample_feature = sample_cams[i].to(device)
                    proposed_feature = proposed_cams[i].to(device)
                    
                    teacher_cam = create_teacher_cam(image, label, teacher_feature, teacher)
                    student_cam = create_student_cam(image, label, student_feature, student)
                    testmodel_cam = create_student_cam(image, label, testmodel_feature, test_model)
                    st_cam = create_student_cam(image, label, st_feature, st)
                    # sample_cam = create_student_cam(image, label, st_feature, sample)
                    proposed_cam = create_student_cam(image, label, proposed_feature, proposed)
                    
                    # 1. 教師と生徒のLoss
                    teacher_student_loss += cam_loss(teacher_cam, student_cam)
                    # 2. 教師とテストモデルのLoss
                    teacher_testmodel_loss += cam_loss(teacher_cam, testmodel_cam)
                    # 3. 教師と従来法のLoss
                    teacher_st_loss += cam_loss(teacher_cam, st_cam)
                    # 4. 教師とサンプルモデルのLoss
                    # teacher_sample_loss += cam_loss(teacher_cam, sample_cam)
                    # 5. 教師と提案法0.1のLoss
                    teacher_proposed_loss += cam_loss(teacher_cam, proposed_cam)
            
    teacher_student_loss /= iteration
    teacher_st_loss /= iteration
    teacher_testmodel_loss /= iteration
    # teacher_sample_loss /= iteration
    teacher_proposed_loss /= iteration
    
    print('Teacher & Student: '         + str(teacher_student_loss.numpy()))
    print('Teacher & Distillation '     + str(teacher_st_loss.numpy()))
    print('Teacher & Proposed: '        + str(teacher_testmodel_loss.numpy()))
    # print('Teacher & only CAMLoss: '    + str(teacher_sample_loss.numpy()))
    print('Teacher & Proposed-0.1: '    + str(teacher_proposed_loss.numpy()))

def create_teacher_cam(image, label, feature, model):
    weight = model.fc.weight[label]
    weight = weight.reshape(1024, 1, 1)
    cam = feature * weight  
    cam = cam.detach().cpu().numpy()
    cam = np.sum(cam, axis=0)
    # cam = cv2.resize(cam, (32, 32))
    # cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = torch.tensor(cam)
    return cam

def create_student_cam(image, label, feature, model):
    weight = model.fc.weight[label]
    weight = weight.reshape(512, 1, 1)
    cam = feature * weight  
    cam = cam.detach().cpu().numpy()
    cam = np.sum(cam, axis=0)
    # cam = cv2.resize(cam, (32, 32))
    # cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = torch.tensor(cam)
    return cam


if __name__ == '__main__':
    main()