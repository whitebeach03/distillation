import cv2
import torch
import numpy as np
from src.model import resnet_student, resnet_teacher, Student, Teacher
from torchvision.models.feature_extraction import create_feature_extractor
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
from src.utils import *

name = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

def main():
    model_type = 'normal'
    seed = 7
    batch_size = 128
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BICUBIC = InterpolationMode.BICUBIC

    # setting models
    if model_type == 'resnet':
        student = resnet_student().to(device)
        teacher = resnet_teacher().to(device)
        cam     = resnet_student().to(device)
        st      = resnet_student().to(device)
    elif model_type == 'normal':
        student = Student().to(device)
        teacher = Teacher().to(device)
        st      = Student().to(device)
        cam05   = Student().to(device)
        cam10   = Student().to(device)

    student.load_state_dict(torch.load('./logs/' + str(model_type) + '/student/200_0.pth'))
    teacher.load_state_dict(torch.load('./logs/' + str(model_type) + '/teacher/200_0.pth'))
    st.load_state_dict(torch.load('logs/' + str(model_type) + '/st/200_0.pth'))
    cam05.load_state_dict(torch.load('logs/' + str(model_type) + '/cam/05_200_0.pth'))
    cam10.load_state_dict(torch.load('logs/' + str(model_type) + '/cam/10_200_0.pth'))
 
    student.eval()
    teacher.eval()
    st.eval()
    cam05.eval()
    cam10.eval()

    # setting dataset
    data_dir        = './data/cifar10'
    input_transform = transforms.Compose([transforms.Resize(32, interpolation=BICUBIC), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_transform   = transforms.Compose([transforms.Resize(32, interpolation=BICUBIC), transforms.ToTensor()])
    dataset         = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=img_transform)
    dataloader      = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        _, student_cams = student(images)
        _, teacher_cams = teacher(images)
        _, st_cams      = st(images)
        _, cam05_cams   = cam05(images)
        _, cam10_cams   = cam10(images)

        # show 10-CAM
        for i in range(10):
            image = images[i]
            label = int(labels[i])
            student_feature = student_cams[i].to(device)
            teacher_feature = teacher_cams[i].to(device)
            st_feature      = st_cams[i].to(device)
            cam05_feature   = cam05_cams[i].to(device)
            cam10_feature   = cam10_cams[i].to(device)
          
            student_cam = create_student_cam(image, label, student_feature, student)
            teacher_cam = create_teacher_cam(image, label, teacher_feature, teacher)
            st_cam      = create_student_cam(image, label, st_feature, st)
            cam05_cam   = create_student_cam(image, label, cam05_feature, cam05)
            cam10_cam   = create_student_cam(image, label, cam10_feature, cam10)
            
            fig, ax = plt.subplots(1, 6)
            ax[0].set_title('Image', size=10)
            ax[1].set_title('Teacher', size=10)
            ax[2].set_title('BP', size=10)
            ax[3].set_title('KD', size=10)
            ax[4].set_title('Prop(0.5)', size=10)
            ax[5].set_title('Prop(0.5->0)', size=10)
            
            ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
            ax[1].imshow(teacher_cam)
            ax[2].imshow(student_cam)
            ax[3].imshow(st_cam)
            ax[4].imshow(cam05_cam)
            ax[5].imshow(cam10_cam)
    
            plt.suptitle(name[label], fontsize=17)
            plt.savefig('./cam/' + str(model_type) + '/' + str(seed) + '_' + str(i) + '.png')
        
        break

def create_student_cam(image, label, feature, student):
    weight = student.fc.weight[label]
    weight = weight.reshape(64, 1, 1)
    cam = feature * weight  
    cam = cam.detach().cpu().numpy()
    cam = np.sum(cam, axis=0)
    cam = cv2.resize(cam, (32, 32))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    
    visualization = show_cam_on_image(image.permute(1, 2, 0).cpu().numpy(), cam, use_rgb=True)
    return visualization

def create_teacher_cam(image, label, feature, teacher):
    weight = teacher.fc.weight[label]
    weight = weight.reshape(256, 1, 1)
    cam = feature * weight  
    cam = cam.detach().cpu().numpy()
    cam = np.sum(cam, axis=0)
    cam = cv2.resize(cam, (32, 32))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    
    visualization = show_cam_on_image(image.permute(1, 2, 0).cpu().numpy(), cam, use_rgb=True)
    return visualization

def create_sample_cam(image, label, feature, proposed):
    cam = 0
    weight = proposed.fc.weight[label]
    for i in range(128):
        cam += feature[i] * weight[i]
        
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (32, 32))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    
    visualization = show_cam_on_image(image.permute(1, 2, 0).cpu().numpy(), cam, use_rgb=True)
    return visualization

if __name__ == '__main__':
    main()