import cv2
import torch
import numpy as np
from src.model import resnet_student, resnet_teacher, SampleModel
from torchvision.models.feature_extraction import create_feature_extractor
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

name = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

def main():
    batch_size = 128
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BICUBIC = InterpolationMode.BICUBIC

    # setting ResNet 
    student = resnet_student().to(device)
    teacher = resnet_teacher().to(device)
    model = resnet_student().to(device)
    # st = resnet_student().to(device)
    

    student.load_state_dict(torch.load('./logs/resnet/student/0.pth'))
    teacher.load_state_dict(torch.load('./logs/resnet/teacher/0.pth'))
    model.load_state_dict(torch.load('logs/resnet/cam/0.pth'))
    # st.load_state_dict(torch.load('logs/resnet/st/0.pth'))
 
    student.eval()
    teacher.eval()
    model.eval()
    # st.eval()

    # setting dataset
    data_dir = './data/cifar10'
    input_transform = transforms.Compose([transforms.Resize(32, interpolation=BICUBIC), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_transform = transforms.Compose([transforms.Resize(32, interpolation=BICUBIC), transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        preds, student_cams = student(images)
        preds, teacher_cams = teacher(images)
        preds, model_cams = model(images)
        # preds, st_cams = st(images)

        # show 10-CAM
        for i in range(10):
            image = images[i]
            pred = preds[i].argmax()
            label = int(labels[i])
            student_feature = student_cams[i].to(device)
            teacher_feature = teacher_cams[i].to(device)
            model_feature = model_cams[i].to(device)
            # st_feature = st_cams[i].to(device)
          
            student_cam = create_student_cam(image, label, student_feature, student)
            teacher_cam = create_teacher_cam(image, label, teacher_feature, teacher)
            model_cam = create_student_cam(image, label, model_feature, model)
            # st_cam = create_student_cam(image, label, st_feature, st)
            
            fig, ax = plt.subplots(1, 4)
            ax[0].set_title('Student')
            ax[1].set_title('Teacher')
            ax[2].set_title('Proposed')
            # ax[3].set_title('Distillation')
            ax[3].set_title('Image')
            ax[0].imshow(student_cam)
            ax[1].imshow(teacher_cam)
            ax[2].imshow(model_cam)
            # ax[3].imshow(st_cam)
            ax[3].imshow(image.permute(1, 2, 0).cpu().numpy())
    
            plt.suptitle(name[label], fontsize=20)
            plt.savefig('./cam/sample_cam' + str(i) + '.png')
        
        break

def create_student_cam(image, label, feature, student):
    weight = student.fc.weight[label]
    weight = weight.reshape(512, 1, 1)
    cam = feature * weight  
    cam = cam.detach().cpu().numpy()
    cam = np.sum(cam, axis=0)
    cam = cv2.resize(cam, (32, 32))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    
    visualization = show_cam_on_image(image.permute(1, 2, 0).cpu().numpy(), cam, use_rgb=True)
    return visualization

def create_teacher_cam(image, label, feature, teacher):
    weight = teacher.fc.weight[label]
    weight = weight.reshape(1024, 1, 1)
    cam = feature * weight  
    cam = cam.detach().cpu().numpy()
    cam = np.sum(cam, axis=0)
    cam = cv2.resize(cam, (32, 32))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    
    visualization = show_cam_on_image(image.permute(1, 2, 0).cpu().numpy(), cam, use_rgb=True)
    return visualization

def create_sample_cam(image, label, feature, model):
    cam = 0
    weight = model.fc.weight[label]
    for i in range(128):
        cam += feature[i] * weight[i]
        
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (32, 32))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    
    visualization = show_cam_on_image(image.permute(1, 2, 0).cpu().numpy(), cam, use_rgb=True)
    return visualization

if __name__ == '__main__':
    main()