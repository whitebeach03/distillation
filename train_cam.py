from tqdm import tqdm
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
import pickle
from src.utils import *

def main():
    for i in range(3, 4):
        print(i)
        model_type = 'normal'
        cam_rate = '10' # default: '01', CAM-curriculum: '10'
        epochs = 200
        batch_size = 128
        # torch.manual_seed(i)
        # np.random.seed(i)
        seed_everything(100+i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data_dir = './data/cifar10'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainset = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=transform)
        testset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
        n_samples = len(trainset)
        n_train = int(n_samples * 0.8)
        n_val = n_samples - n_train
        trainset, valset = random_split(trainset, [n_train, n_val])
        
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        if model_type == 'resnet':
            teacher = resnet_teacher().to(device)
            student = resnet_student().to(device)
        elif model_type == 'normal':
            teacher = Teacher().to(device)
            student = Student().to(device)
        
        teacher.load_state_dict(torch.load('./logs/' + str(model_type) + '/teacher/' + str(epochs) + '_' + str(i) + '.pth')) 
        loss_fn = nn.CrossEntropyLoss() 
        student_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # train
        optim = optimizers.Adam(student.parameters())
        T = 10 
        score = 0.
        soft_loss = SoftTargetLoss() # Soft Loss
        cam_loss = nn.MSELoss() # CAM Loss
        for epoch in range(epochs):
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            student.train()
            for cnt, (images, labels) in enumerate(tqdm(train_dataloader, leave=False)):
                images, labels = images.to(device), labels.to(device)
                preds, student_features = student(images)
                targets, teacher_features = teacher(images)
                
                if cam_rate != '10':
                    student_cam = create_student_cam(student, images, labels, student_features, batch_size, device)
                    teacher_cam = create_teacher_cam(teacher, images, labels, teacher_features, batch_size, device)
                    camloss = cam_loss(student_cam, teacher_cam)
                    if cam_rate == '00':
                        loss = camloss
                    elif cam_rate == '01':
                        loss = 0.5*loss_fn(preds, labels) + 0.4*T*T*soft_loss(preds, targets) + 0.1*camloss
                    elif cam_rate == '02': 
                        loss = 0.5*loss_fn(preds, labels) + 0.3*T*T*soft_loss(preds, targets) + 0.2*camloss
                    elif cam_rate == '03':
                        loss = 0.5*loss_fn(preds, labels) + 0.2*T*T*soft_loss(preds, targets) + 0.3*camloss
                    elif cam_rate == '04':
                        loss = 0.5*loss_fn(preds, labels) + 0.1*T*T*soft_loss(preds, targets) + 0.4*camloss
                    elif cam_rate == '05':
                        loss = 0.5*loss_fn(preds, labels) + 0.5*camloss
                # CAM Curriculum learning
                elif cam_rate == '10':     
                    if epoch <= 20:
                        student_cam = create_student_cam(student, images, labels, student_features, batch_size, device)
                        teacher_cam = create_teacher_cam(teacher, images, labels, teacher_features, batch_size, device)
                        # loss = 0.5*loss_fn(preds, labels) + 0.4*T*T*soft_loss(preds, targets) + 0.1*cam_loss(student_cam, teacher_cam)
                        loss = 0.5*loss_fn(preds, labels) + 0.5*cam_loss(student_cam, teacher_cam)
                    else:
                        loss = 0.5*loss_fn(preds, labels) + 0.5*T*T*soft_loss(preds, targets)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss += loss.item()
                train_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
                # camloss += loss.item()
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            # camloss /= len(train_dataloader)
            # student_hist['cam_loss'].append(camloss)
            
            # student validation
            student.eval()
            with torch.no_grad():
                for (images, labels) in val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    preds, _ = student(images)
                    targets, _ = teacher(images)
                    loss = 0.5*loss_fn(preds, labels) + 0.5*T*T*soft_loss(preds, targets)
                    val_loss += loss.item()
                    val_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
                
            if score <= val_acc:
                print('save param')
                score = val_acc
                torch.save(student.state_dict(), './logs/' + str(model_type) + '/cam/' + str(cam_rate) + '_' + str(epochs) + '_' + str(i) + '.pth') 
            
            student_hist['loss'].append(train_loss)
            student_hist['accuracy'].append(train_acc)
            student_hist['val_loss'].append(val_loss)
            student_hist['val_accuracy'].append(val_acc)

            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            
            with open('./history/' + str(model_type) + '/cam/' + str(cam_rate) + '_' + str(epochs) + '_' + str(i) + '.pickle', mode='wb') as f: 
                pickle.dump(student_hist, f)
        
        student.load_state_dict(torch.load('./logs/' + str(model_type) + '/cam/' + str(cam_rate) + '_' + str(epochs) + '_' + str(i) + '.pth')) 
        test = {'acc': [], 'loss': []}
        # distillation student test
        student.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (images,labels) in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                preds, _ = student(images)
                loss = loss_fn(preds, labels)
                test_loss += loss.item()
                test_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        test['acc'].append(test_acc)
        test['loss'].append(test_loss)
        
        with open('./history/' + str(model_type) + '/cam/' + str(cam_rate) + '_' + str(epochs) + '_test' + str(i) + '.pickle', mode='wb') as f: # 変更箇所
            pickle.dump(test, f)

def create_student_cam(model, images, labels, features, batch_size, device):
    attmap = torch.tensor([]).to(device)
    for i in range(batch_size):
        image = images[i]
        feature = features[i]
        
        for j in range(10):
            weight = model.fc.weight[j]
            weight = weight.reshape(64, 1, 1) # (64, 1, 1)
            cam = feature * weight
            cam = torch.sum(cam, axis=0)    
            attmap = torch.cat((attmap, cam), dim=0)
    return attmap

def create_teacher_cam(model, images, labels, features, batch_size, device):
    attmap = torch.tensor([]).to(device)
    for i in range(batch_size):
        image = images[i]
        feature = features[i]
   
        for j in range(10):
            weight = model.fc.weight[j]
            weight = weight.reshape(256, 1, 1) # (256, 1, 1)
            cam = feature * weight
            cam = torch.sum(cam, axis=0)
            attmap = torch.cat((attmap, cam), dim=0)
    return attmap



if __name__ == '__main__':
    main()