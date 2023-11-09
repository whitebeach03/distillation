import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from src.model import Model, TModel
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

name = {0: 'covid-19', 1: 'normal', 2: 'opacity', 3: 'pneumonia'}

def main():
    BICUBIC = InterpolationMode.BICUBIC
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting models
    student = Model().to(device)
    teacher = TModel().to(device)
    st = Model().to(device)
    
    student.load_state_dict(torch.load('./logs/student/00.pth'))
    teacher.load_state_dict(torch.load('./logs/teacher/00.pth'))
    st.load_state_dict(torch.load('./logs/student_st/00.pth'))
    
    student.eval()
    teacher.eval()
    st.eval()

    # setting dataset
    data_dir = './covid19'
    input_transform = transforms.Compose([transforms.Resize(224, interpolation=BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_transform = transforms.Compose([transforms.Resize(224, interpolation=BICUBIC), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=data_dir)

    for i in range(10):
        image, label = dataset[i]
        input_image = input_transform(image)
        image = img_transform(image)

        # create CAM
        student_cam = create_cam(student, input_image, image, label)
        teacher_cam = create_cam(teacher, input_image, image, label)
        st_cam = create_cam(st, input_image, image, label)
    
        # visualize and save CAM
        fig, ax = plt.subplots(1, 3)
        ax[0].set_title('student')
        ax[1].set_title('teacher')
        ax[2].set_title('distillation')
    
        ax[0].imshow(student_cam)
        ax[1].imshow(teacher_cam)
        ax[2].imshow(st_cam)
        
        plt.suptitle(name[label], fontsize=20)
        plt.savefig('./cam/cam_0' + str(i) + '.png')

def create_cam(model, input_image, image, label):
    target_layers = [model.layer4]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_image.unsqueeze(0), targets=[ClassifierOutputTarget(label)])
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
    return visualization

if __name__ =='__main__':
    main()
