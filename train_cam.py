from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from src.model import TeacherModel, StudentModel
from src.utils import EarlyStopping
import torch.optim as optimizers
from sklearn.metrics import accuracy_score
from src.kd_loss.st import SoftTargetLoss
from src.kd_loss.cam_loss import CAMLoss
import pickle
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM

def main():
    for i in range(1):
        np.random.seed(i)
        torch.manual_seed(i)
        batch_size = 128
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data_dir = './data/cifar10'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        trainset = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=transform)
        testset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
        
        n_samples = len(trainset)
        n_train = int(n_samples * 0.8)
        n_val = n_samples - n_train
        trainset, valset = random_split(trainset, [n_train, n_val])
        
        train_dataloader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True, num_workers=8)
        val_dataloader = DataLoader(valset, batch_size=128, shuffle=False)
        test_dataloader = DataLoader(testset, batch_size=128, shuffle=False)
        
        teacher = TeacherModel().to(device)
        student = StudentModel().to(device)
        
        teacher.load_state_dict(torch.load('./logs/teacher/' + str(i) + '.pth'))
        
        loss_fn = nn.CrossEntropyLoss()
        
        epochs = 100
        student_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # teacher test
        teacher.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (images,labels) in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                preds = teacher(images)
                loss = loss_fn(preds, labels)
                test_loss += loss.item()
                test_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        # test_loss: 0.662, test_accuracy: 0.819
        
        # train
        optim = optimizers.Adam(student.parameters())
        T = 10 # 温度パラメータ
        score = 0.
        soft_loss = SoftTargetLoss() # ソフトターゲット
        # cam_loss = nn.MSELoss() # CAMターゲット
        cam_loss = CAMLoss()
        for epoch in range(epochs):
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            student.train()
            for cnt, (images, labels) in enumerate(tqdm(train_dataloader, leave=False)):
                images, labels = images.to(device), labels.to(device)
                preds = student(images)
                targets = teacher(images)
                
                if cnt % 10 == 0:
                    student_cam = cam(student, images, labels, batch_size, device)
                    teacher_cam = cam(teacher, images, labels, batch_size, device) # torch.Size([batch_size=128, 32, 32])
                    loss = loss_fn(preds, labels) + T*T*soft_loss(preds, targets) + 0.1*cam_loss(student_cam, teacher_cam, batch_size)
                else:
                    loss = loss_fn(preds, labels) + T*T*soft_loss(preds, targets)
                    
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
                    preds = student(images)
                    targets= teacher(images)
                    loss = loss_fn(preds, labels) + T * T * soft_loss(preds, targets)
                    val_loss += loss.item()
                    val_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
                
            if score <= val_acc:
                print('test')
                score = val_acc
                torch.save(student.state_dict(), './logs/student_cam/' + str(i) + '.pth') 
            
            student_hist['loss'].append(train_loss)
            student_hist['accuracy'].append(train_acc)
            student_hist['val_loss'].append(val_loss)
            student_hist['val_accuracy'].append(val_acc)

            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            
            with open('./history/student_cam/sample' + str(i) + '.pickle', mode='wb') as f:
                pickle.dump(student_hist, f)
        
        student.load_state_dict(torch.load('./logs/student_cam/' + str(i) + '.pth'))
        test = {'acc': [], 'loss': []}
        # distillation student test
        student.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (images,labels) in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                preds = student(images)
                loss = loss_fn(preds, labels)
                test_loss += loss.item()
                test_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        test['acc'].append(test_acc)
        test['loss'].append(test_loss)
        with open('./history/student_cam/test' + str(i) + '.pickle', mode='wb') as f:
            pickle.dump(test, f)

def cam(model, images, labels, batch_size, device):
    model.eval()
    target_layers = [model.layer4]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    cams = np.array([])
    
    for i in range(batch_size):
        image = images[i]
        label = labels[i]
        grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=[ClassifierOutputTarget(label)])
        grayscale_cam = grayscale_cam[0, :]
        cams = np.append(cams, grayscale_cam).reshape(i+1, 32, 32)
    
    cams = torch.tensor(cams)
    return cams

if __name__ == '__main__':
    main()