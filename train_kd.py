from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from src.model import TeacherModel, StudentModel, TModel, Model
from src.utils import EarlyStopping
import torch.optim as optimizers
from sklearn.metrics import accuracy_score
from src.kd_loss.st import SoftTargetLoss
import pickle

def main():
    for i in range(1, 5):
        print(i+1)
        epochs = 50
        batch_size = 32
        np.random.seed(i)
        torch.manual_seed(i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # data_dir = './data/cifar10'
        # transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        # trainset = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=transform)
        # testset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
        
        data_dir = './covid19'
        transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        
        n_samples = len(dataset)
        n_val = int(n_samples * 0.2)
        n_test = n_val
        n_train = n_samples - n_val - n_test
        trainset, valset, testset = random_split(dataset, [n_train, n_val, n_test])
        
        # n_samples = len(trainset)
        # n_train = int(n_samples * 0.8)
        # n_val = n_samples - n_train
        # trainset, valset = random_split(trainset, [n_train, n_val])
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        teacher = TModel().to(device)
        student = Model().to(device)
        
        teacher.load_state_dict(torch.load('./logs/teacher/0' + str(i) + '.pth'))
        loss_fn = nn.CrossEntropyLoss()
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

        optim = optimizers.Adam(student.parameters())
        T = 10 #温度パラメータ
        score = 0.
        soft_loss = SoftTargetLoss()
        for epoch in range(epochs):
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            # train
            student.train()
            for (images, labels) in tqdm(train_dataloader, leave=False):
                images, labels = images.to(device), labels.to(device)
                preds = student(images)
                targets = teacher(images)
                loss = loss_fn(preds, labels) + T*T*soft_loss(preds,targets)
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
                    loss = loss_fn(preds, labels) + T*T*soft_loss(preds, targets)
                    val_loss += loss.item()
                    val_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
                
            if score <= val_acc:
                print('test')
                score = val_acc
                torch.save(student.state_dict(), './logs/student_st/0' + str(i) + '.pth') 
            
            student_hist['loss'].append(train_loss)
            student_hist['accuracy'].append(train_acc)
            student_hist['val_loss'].append(val_loss)
            student_hist['val_accuracy'].append(val_acc)

            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            
            with open('./history/student_st/0' + str(i) + '.pickle', mode='wb') as f:
                pickle.dump(student_hist, f)

        student.load_state_dict(torch.load('./logs/student_st/0' + str(i) + '.pth'))
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
        # test_loss: 0.825, test_accuracy: 0.811
        test['acc'].append(test_acc)
        test['loss'].append(test_loss)
        with open('./history/student_st/test0'+str(i)+'.pickle', mode='wb') as f:
            pickle.dump(test, f)

if __name__ == '__main__':
    main()