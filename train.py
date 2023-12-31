from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from src.model import resnet_student, resnet_teacher, Student, Teacher
from src.utils import EarlyStopping
import torch.optim as optimizers
from sklearn.metrics import accuracy_score
from src.kd_loss.st import SoftTargetLoss
from src.utils import *
import pickle

def main():
    for i in range(5):
        print(i)
        model_type = 'resnet'
        model_size = 'student'
        epochs = 150
        batch_size = 128
        # torch.manual_seed(i)
        # np.random.seed(i)
        seed_everything(100+i)
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        es = EarlyStopping(patience=5, verbose=1)
        
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
        
        # setting model
        if model_type == 'resnet':
            if model_size == 'teacher':
                model = resnet_teacher().to(device)
            elif model_size == 'student':
                model = resnet_student().to(device)
        elif model_type == 'normal':
            if model_size == 'teacher':
                model = Teacher().to(device)
            elif model_size == 'student':
                model = Student().to(device)

        optim = optimizers.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        score = 0.
        history = {'loss': [], 'accuracy': [], 'val_loss':[], 'val_accuracy': []}
        
        for epoch in range(epochs):
            train_acc = 0.
            train_loss = 0.
            val_acc = 0.
            val_loss = 0.
            
            # train
            model.train()
            for (images, labels) in tqdm(train_dataloader, leave=False):
                images, labels = images.to(device), labels.to(device)
                
                preds, _ = model(images)
                loss = loss_fn(preds,labels)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                train_loss += loss.item()
                train_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
                
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            
            # validation
            model.eval()
            with torch.no_grad():
                for (images, labels) in val_dataloader:
                    images,labels = images.to(device), labels.to(device)
                    
                    preds, _ = model(images)
                    loss = loss_fn(preds, labels)
                    
                    val_loss += loss.item()
                    val_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
                    
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
            
            # save params
            if score <= val_acc:
                print('save param')
                score = val_acc
                torch.save(model.state_dict(), './logs/' + str(model_type) + '/' + str(model_size) + '/' + str(epochs) + '_' + str(i) + '.pth') 

            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
        
        with open('./history/' + str(model_type) + '/' + str(model_size) + '/' + str(epochs) + '_' + str(i) + '.pickle', mode='wb') as f: 
            pickle.dump(history, f)

        # model test
        model.load_state_dict(torch.load('./logs/' + str(model_type) + '/' + str(model_size) + '/' + str(epochs) + '_' + str(i) + '.pth'))
        model.eval()
        
        test = {'acc': [], 'loss': []}
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (images,labels) in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                
                preds, _ = model(images)
                loss = loss_fn(preds, labels)
                
                test_loss += loss.item()
                test_acc += accuracy_score(labels.tolist(), preds.argmax(dim=-1).tolist())
                
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')

        test['acc'].append(test_acc)
        test['loss'].append(test_loss)
        
        with open('./history/' + str(model_type) + '/' + str(model_size) + '/' + str(epochs) + '_' + 'test' + str(i) + '.pickle', mode='wb') as f: 
            pickle.dump(test, f)
        
    
if __name__ == '__main__':
    main()