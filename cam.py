import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from src.model import TeacherModel, StudentModel, Model
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

name = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

def main():
    BICUBIC = InterpolationMode.BICUBIC
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting models
    student = StudentModel().to(device)
    teacher = TeacherModel().to(device)
    st_student = StudentModel().to(device)
    cam_student = StudentModel().to(device)
    model = Model().to(device)
    
    # student.load_state_dict(torch.load('./logs/student/0.pth'))
    # teacher.load_state_dict(torch.load('./logs/teacher/0.pth'))
    # st_student.load_state_dict(torch.load('./logs/student_st/0.pth'))
    # cam_student.load_state_dict(torch.load('./logs/student_cam/0.pth'))
    model.load_state_dict(torch.load('./logs/student/00.pth'))
    
    student.eval()
    teacher.eval()
    st_student.eval()
    cam_student.eval()
    model.eval()

    # setting dataset
    # data_dir = './data/cifar10'
    # input_transform = transforms.Compose([transforms.Resize(32, interpolation=BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # img_transform = transforms.Compose([transforms.Resize(32, interpolation=BICUBIC), transforms.ToTensor()])
    # dataset = datasets.CIFAR10(root=data_dir, download=True, train=True)
    
    train_data_dir = './covid19/train'
    test_data_dir = './covid19/test'
    input_transform = transforms.Compose([transforms.Resize(224, interpolation=BICUBIC), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_transform = transforms.Compose([transforms.Resize(224, interpolation=BICUBIC), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=train_data_dir)######

    for i in range(2):
        image, label = dataset[i]
        input_image = input_transform(image)
        image = img_transform(image)

        # create CAM
        # student_cam = create_cam(student, input_image, image, label)
        # teacher_cam = create_cam(teacher, input_image, image, label)
        # st_student_cam = create_cam(st_student, input_image, image, label)
        # cam_student_cam = create_cam(cam_student, input_image, image, label)
        model_cam = create_cam(model, input_image, image, label)

        # visualize and save CAM
        fig, ax = plt.subplots(1)
        # ax[0].set_title('student')
        # ax[1].set_title('teacher')
        # ax[2].set_title('student(distilled)')
        # ax[3].set_title('student(CAM)')
        
        # ax[0].imshow(student_cam)
        # ax[1].imshow(teacher_cam)
        # ax[2].imshow(st_student_cam)
        # ax[3].imshow(cam_student_cam)
        ax.imshow(model_cam)
        
        # plt.suptitle(name[label], fontsize=20)
        # plt.savefig('./cam/cam_' + str(i) + '.png')
        
        plt.savefig('a.png')

def create_cam(model, input_image, image, label):
    target_layers = [model.layer4]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_image.unsqueeze(0), targets=[ClassifierOutputTarget(label)])
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
    return visualization

if __name__ =='__main__':
    main()
