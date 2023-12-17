from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from src.model import resnet_student, resnet_teacher
from src.utils import *
import torch.optim as optimizers
from sklearn.metrics import accuracy_score
from src.kd_loss.st import SoftTargetLoss
import pickle

def main():
    for i in range(6, 8):
        print(i)
        epochs = 150
        batch_size = 128
        # torch.manual_seed(i)
        # np.random.seed(i)
        seed_everything(100+i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data_dir = './data/cifar10'
        transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
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
        # teacher = Teacher().to(device)
        # student = Student().to(device)
        
        teacher.load_state_dict(torch.load('./logs/resnet/teacher/' + str(epochs) + '_' + str(i) + '.pth')) 
        # teacher.load_state_dict(torch.load('./logs/teacher/' + str(epochs) + '_' + str(i) + '.pth')) 
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

        optim = optimizers.Adam(student.parameters())
        T = 10 
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
                preds, _ = student(images)
                targets, _ = teacher(images)
                loss = 0.5*loss_fn(preds, labels) + 0.5*T*T*soft_loss(preds, targets)
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
                torch.save(student.state_dict(), './logs/resnet/st/' + str(epochs) + '_' + str(i) + '.pth') 
                # torch.save(student.state_dict(), './logs/st/' + str(epochs) + '_' + str(i) + '.pth') 
            
            student_hist['loss'].append(train_loss)
            student_hist['accuracy'].append(train_acc)
            student_hist['val_loss'].append(val_loss)
            student_hist['val_accuracy'].append(val_acc)

            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            
            with open('./history/resnet/st/' + str(epochs) + '_' + str(i) + '.pickle', mode='wb') as f:
                pickle.dump(student_hist, f)
            # with open('./history/st/' + str(epochs) + '_' + str(i) + '.pickle', mode='wb') as f:
            #     pickle.dump(student_hist, f)

        student.load_state_dict(torch.load('./logs/resnet/st/' + str(epochs) + '_' + str(i) + '.pth'))
        # student.load_state_dict(torch.load('./logs/st/' + str(epochs) + '_' + str(i) + '.pth'))
        test = {'acc': [], 'loss': []}
        # distillation student test
        student.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (images, labels) in test_dataloader:
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
        with open('./history/resnet/st/' + str(epochs) + '_' + 'test' + str(i) + '.pickle', mode='wb') as f:
            pickle.dump(test, f)
        # with open('./history/st/' + str(epochs) + '_' + 'test' + str(i) + '.pickle', mode='wb') as f:
        #     pickle.dump(test, f)
        
if __name__ == '__main__':
    main()