import cv2
import torch
import numpy as np
from src.model import resnet
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import torchvision

batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setting ResNet model
model = resnet().to(device)
model.load_state_dict(torch.load('./logs/resnet/0.pth'))
model.eval()

# setting dataset
data_dir = './data/cifar10'
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for images, labels in dataloader:
#     images, labels = images.to(device), labels.to(device)
#     extractor = create_feature_extractor(model, ['conv5_x'])
#     features = extractor(images)['conv5_x']
    
#     for i in range(batch_size):
#         cam = 0
#         image = images[i]
#         label = labels[i]
#         feature = features[i].to(device)
#         weight = model.fc.weight[label]
#         for j in range(512):
#             cam += feature[j] * weight[j]
        
#         cam = cam.detach().cpu().numpy()
#         cam = cv2.resize(cam, (32, 32))
#         cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
#         cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#         cv2.imwrite("result/resnet_cam.png", cam)
    
#         break
#     break

image, label = dataset[60]
image = torchvision.transforms.functional.to_pil_image(image)
print(type(image))
imgPIL = Image.open(image)  # 画像読み込み

imgPIL.show()  # 画像表示