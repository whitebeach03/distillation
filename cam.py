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
    model_type = 'resnet'
    seed = 9
    batch_size = 32
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BICUBIC = InterpolationMode.BICUBIC

    # setting models
    if model_type == 'resnet':
        student = resnet_student().to(device)
        teacher = resnet_teacher().to(device)
        st      = resnet_student().to(device)
        cam1    = resnet_student().to(device)
        cam2    = resnet_student().to(device)
        cam3    = resnet_student().to(device)
        cam4    = resnet_student().to(device)
        cam5    = resnet_student().to(device)
    elif model_type == 'normal':
        student = Student().to(device)
        teacher = Teacher().to(device)
        st      = Student().to(device)
        cam05   = Student().to(device)
        cam10   = Student().to(device)

    student.load_state_dict(torch.load('./logs/' + str(model_type) + '/student/150_0.pth'))
    teacher.load_state_dict(torch.load('./logs/' + str(model_type) + '/teacher/150_0.pth'))
    st.load_state_dict(torch.load('./logs/' + str(model_type) + '/st/150_0.pth'))
    cam1.load_state_dict(torch.load('./logs/' + str(model_type) + '/cam/01_150_0.pth'))
    cam2.load_state_dict(torch.load('./logs/' + str(model_type) + '/cam/02_150_0.pth'))
    cam3.load_state_dict(torch.load('./logs/' + str(model_type) + '/cam/03_150_0.pth'))
    cam4.load_state_dict(torch.load('./logs/' + str(model_type) + '/cam/04_150_0.pth'))
    cam5.load_state_dict(torch.load('./logs/' + str(model_type) + '/cam/05_150_0.pth'))
 
    student.eval()
    teacher.eval()
    st.eval()
    cam1.eval()
    cam2.eval()
    cam3.eval()
    cam4.eval()
    cam5.eval()

    # setting dataset
    data_dir        = './data/cifar10'
    input_transform = transforms.Compose([transforms.Resize(32, interpolation=BICUBIC), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_transform   = transforms.Compose([transforms.Resize(32, interpolation=BICUBIC), transforms.ToTensor()])
    dataset         = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=img_transform)
    dataloader      = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for images, labels in dataloader:
        images, labels  = images.to(device), labels.to(device)
        _, student_cams = student(images)
        _, teacher_cams = teacher(images)
        _, st_cams      = st(images)
        _, cam1_cams    = cam1(images)
        _, cam2_cams    = cam2(images)
        _, cam3_cams    = cam3(images)
        _, cam4_cams    = cam4(images)
        _, cam5_cams    = cam5(images)

        # show 10-CAM
        for i in range(10):
            image = images[i]
            label = int(labels[i])
            student_feature = student_cams[i].to(device)
            teacher_feature = teacher_cams[i].to(device)
            st_feature      = st_cams[i].to(device)
            cam1_feature    = cam1_cams[i].to(device)
            cam2_feature    = cam2_cams[i].to(device)
            cam3_feature    = cam3_cams[i].to(device)
            cam4_feature    = cam4_cams[i].to(device)
            cam5_feature    = cam5_cams[i].to(device)
          
            student_cam = create_student_cam(image, label, student_feature, student)
            teacher_cam = create_teacher_cam(image, label, teacher_feature, teacher)
            st_cam      = create_student_cam(image, label, st_feature, st)
            cam1_cam    = create_student_cam(image, label, cam1_feature, cam1)
            cam2_cam    = create_student_cam(image, label, cam2_feature, cam2)
            cam3_cam    = create_student_cam(image, label, cam3_feature, cam3)
            cam4_cam    = create_student_cam(image, label, cam4_feature, cam4)
            cam5_cam    = create_student_cam(image, label, cam5_feature, cam5)
            
            fig, ax = plt.subplots(1, 10)
            ax[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax[3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax[4].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax[5].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax[6].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax[7].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax[8].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            ax[9].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            
            ax[0].axis('off')
            ax[1].set_title('Image', size=9)
            ax[2].set_title('Teacher', size=9)
            ax[3].set_title('BP', size=9)
            ax[4].set_title('KD', size=9)
            ax[5].set_title('Prop.1', size=9)
            ax[6].set_title('Prop.2', size=9)
            ax[7].set_title('Prop.3', size=9)
            ax[8].set_title('Prop.4', size=9)
            ax[9].set_title('Prop.5', size=9)
            
            ax[0].text(0.15, 0.48, name[label])
            # ax[0].text(-0.75, 0.48, name[label])
            ax[1].imshow(image.permute(1, 2, 0).cpu().numpy())
            ax[2].imshow(teacher_cam)
            ax[3].imshow(student_cam)
            ax[4].imshow(st_cam)
            ax[5].imshow(cam1_cam)
            ax[6].imshow(cam2_cam)
            ax[7].imshow(cam3_cam)
            ax[8].imshow(cam4_cam)
            ax[9].imshow(cam5_cam)
    
            plt.savefig('./cam/' + str(model_type) + '/' + str(seed) + '_' + str(i) + '.png')
        
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