import cv2
import torch
import torch.nn as nn
import numpy as np
from src.model import SampleModel
from torchvision.models.feature_extraction import create_feature_extractor
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

classes_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

batch_size = 128
data_dir = './data/cifar10'
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
dataset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SampleModel().to(device)
model.load_state_dict(torch.load('./logs/sample/0.pth'))
model.eval()
softmax = nn.Softmax(dim=1)

def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)

with torch.no_grad():
    for (images, labels) in dataloader:
        images, labels = images.to(device), labels .to(device)
        preds, attention = model(images)
        preds = softmax(preds)
        
        # d_inputs = images.data.cpu().numpy()
        # item_img = d_inputs[0]
        # v_img = ((item_img.transpose((1, 2, 0)) * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255
        # v_img = np.uint8(v_img)

        image = images[0]
        label = labels[0]
        features, weights = attention
        feature = features[0]
        weight = weights[label]
        weight = weight.reshape(128, 1, 1)
        attmap = feature * weight
        attmap = attmap.cpu().numpy()
        attmap = np.sum(attmap, axis=0)
        attmap = cv2.resize(attmap, (32, 32))
        attmap = min_max(attmap)
        attmap *= 255
        attmap = np.uint8(attmap)
        print(attmap)
        visualization = cv2.applyColorMap(attmap, cv2.COLORMAP_JET)
        image = image.cpu().numpy()
        v_img = np.uint8(image)
        visualization = cv2.addWeighted(v_img, 0.6, attmap, 0.4, 0)
        fig, ax = plt.subplots(1)
        ax.imshow(visualization)
        plt.suptitle(classes_list[label], fontsize=20)
        plt.savefig('./cam/****.png')
        break
        