from tqdm import tqdm
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
import pickle
from torchvision.models.feature_extraction import create_feature_extractor

def main():
    for i in range(1, 5):
        epochs = 50
        batch_size = 128
        np.random.seed(i)
        torch.manual_seed(i)
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
        
        teacher = resnet_teacher().to(device)
        student = resnet_student().to(device)
        
        teacher.load_state_dict(torch.load('./logs/resnet/teacher/' + str(i) + '.pth'))
        loss_fn = nn.CrossEntropyLoss() 
        student_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # teacher test
        teacher.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (images,labels) in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                preds, _ = teacher(images)
                loss = loss_fn(preds, labels)
                test_loss += loss.item()
                test_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        
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
        
                student_cam = create_student_cam(student, images, labels, student_features, batch_size, device)
                teacher_cam = create_teacher_cam(teacher, images, labels, teacher_features, batch_size, device)

                loss = loss_fn(preds, labels) + T*T*soft_loss(preds, targets) + cam_loss(student_cam, teacher_cam)
                    
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss += loss.item()
                train_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            
            # student validation
            student.eval()
            with torch.no_grad():
                for (images,labels) in val_dataloader:
                    images,labels = images.to(device),labels.to(device)
                    preds, _ = student(images)
                    targets, _ = teacher(images)
                    loss = loss_fn(preds, labels) + T*T*soft_loss(preds, targets)
                    val_loss += loss.item()
                    val_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
                
            if score <= val_acc:
                print('test')
                score = val_acc
                torch.save(student.state_dict(), './logs/resnet/cam/' + str(i) + '.pth') 
            
            student_hist['loss'].append(train_loss)
            student_hist['accuracy'].append(train_acc)
            student_hist['val_loss'].append(val_loss)
            student_hist['val_accuracy'].append(val_acc)

            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            
            with open('./history/resnet/cam/' + str(i) + '.pickle', mode='wb') as f:
                pickle.dump(student_hist, f)
        
        student.load_state_dict(torch.load('./logs/resnet/cam/' + str(i) + '.pth'))
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
        with open('./history/resnet/cam/test' + str(i) + '.pickle', mode='wb') as f:
            pickle.dump(test, f)

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
    
        np.append(attmap, cam)
    attmap = torch.tensor(attmap)
    return attmap

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
    
        np.append(attmap, cam)
    attmap = torch.tensor(attmap)
    return attmap

if __name__ == '__main__':
    main()