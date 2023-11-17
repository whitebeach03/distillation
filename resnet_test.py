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
    for i in range(1):
        batch_size = 128
        np.random.seed(i)
        torch.manual_seed(i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data_dir = './data/cifar10'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        teacher = resnet_teacher().to(device)
        student = resnet_student().to(device)
        test_model = resnet_student().to(device)
        
        teacher.load_state_dict(torch.load('./logs/resnet/teacher/' + str(i) + '.pth'))
        student.load_state_dict(torch.load('./logs/resnet/student/' + str(i) + '.pth'))
        test_model.load_state_dict(torch.load('./logs/resnet/cam/' + str(i) + '.pth'))
        
        loss_fn = nn.CrossEntropyLoss()
        cam_loss = nn.MSELoss()
        
        teacher.eval()
        student.eval()
        test_model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, leave=False):
                images, labels = images.to(device), labels.to(device)
                _, teacher_features = teacher(images)
                _, student_features = student(images)
                _, testmodel_features = test_model(images)
                
                teacher_cam = create_teacher_cam(teacher, images, labels, teacher_features, batch_size, device)
                student_cam = create_student_cam(student, images, labels, student_features, batch_size, device)
                testmodel_cam = create_student_cam(test_model, images, labels, testmodel_features, batch_size, device)
                
                teacher_cam_inf = resize_cam(teacher_cam, batch_size)
                student_cam_inf = resize_cam(student_cam, batch_size)
                testmodel_cam_inf = resize_cam(testmodel_cam, batch_size)
                
                # ここに各モデルのCAM誤差計算を記述. 
                # 1. 教師と生徒のLoss
                # 2. 教師とテストモデルのLoss
                
                break
                
                


def create_teacher_cam(model, images, labels, features, batch_size, device):
    attmap = np.array([])
    for i in range(batch_size):
        image, label = images[i], labels[i]
        feature = features[i].to(device)
        
        weight = model.fc.weight[label]
        weight = weight.reshape(1024, 1, 1)
        cam = feature * weight  
        cam = cam.detach().cpu().numpy()
        cam = np.sum(cam, axis=0)
    
        attmap = np.append(attmap, cam)
    attmap = torch.tensor(attmap)
    attmap = attmap.to(device)
    return attmap

def create_student_cam(model, images, labels, features, batch_size, device):
    attmap = np.array([])
    for i in range(batch_size):
        image, label = images[i], labels[i]
        feature = features[i].to(device)
        
        weight = model.fc.weight[label]
        weight = weight.reshape(512, 1, 1)
        cam = feature * weight  
        cam = cam.detach().cpu().numpy()
        cam = np.sum(cam, axis=0)
    
        attmap = np.append(attmap, cam)
    attmap = torch.tensor(attmap)
    attmap = attmap.to(device)
    return attmap

def resize_cam(cams, batch_size):
    cam_array = np.array([])
    for i in range(batch_size):    
        cam = cams[i]   
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (32, 32))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam_array = np.append(cam_array, cam)
        
    cam_array = torch.tensor(cam_array)
    return cam_array

if __name__ == '__main__':
    main()