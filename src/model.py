import torch
import torch.nn as nn

### Normal ###
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        self.cam = x
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.cam

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        self.cam = x
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.cam
        

### ResNet ###
class BasicBlock(nn.Module):
    rate=1
    def __init__(self, input_dim, output_dim, stride=1, drop=0.3):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.residual_function = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(drop),
            self.relu,
            nn.Conv2d(output_dim, output_dim * BasicBlock.rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim * BasicBlock.rate),
            nn.Dropout(drop)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or input_dim != output_dim * BasicBlock.rate:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_dim, output_dim * BasicBlock.rate, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_dim * BasicBlock.rate)
            )
    
    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    rate = 4
    def __init__(self, input_dim, output_dim, stride=1, drop=0.3):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.residual_function = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(drop),
            self.relu,
            nn.Conv2d(output_dim, output_dim, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(drop),
            self.relu,
            nn.Conv2d(output_dim, output_dim * BottleNeck.rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim * BottleNeck.rate),
            nn.Dropout(drop),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or input_dim != output_dim * BottleNeck.rate:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_dim, output_dim * BottleNeck.rate, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_dim * BottleNeck.rate)
            )
    
    def forward(self, x):
        return self.relu((self.residual_function(x) + self.shortcut(x)))

class ResNetStudent(nn.Module):
    def __init__(self, block, num_block, num_classes=10, drop=0.3):
        super().__init__()
        self.input_dim = 16
        self.drop = drop
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1) #3
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2) #4
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2) #6
        self.conv5_x = self._make_layer(block, 128, num_block[3], 2) #3
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(128 * block.rate, num_classes)
    
    def _make_layer(self, block, output_dim, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.input_dim, output_dim, stride, self.drop))
            self.input_dim = output_dim * block.rate
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        self.cam = x
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.cam

class ResNetAssistant(nn.Module):
    def __init__(self, block, num_block, num_classes=10, drop=0.3):
        super().__init__()
        
        self.input_dim = 32
        self.drop = drop
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = self._make_layer(block, 32, num_block[0], 1) #3
        self.conv3_x = self._make_layer(block, 64, num_block[1], 2) #4
        self.conv4_x = self._make_layer(block, 128, num_block[2], 2) #6
        self.conv5_x = self._make_layer(block, 256, num_block[3], 2) #3
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(256 * block.rate, num_classes)
    
    def _make_layer(self, block, output_dim, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.input_dim, output_dim, stride, self.drop))
            self.input_dim = output_dim * block.rate
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        self.cam = x
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.cam

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10, drop=0.3):
        super().__init__()
        self.input_dim = 64
        self.drop = drop
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1) #3
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2) #4
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2) #6
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2) #3
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(512 * block.rate, num_classes)
    
    def _make_layer(self, block, output_dim, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.input_dim, output_dim, stride, self.drop))
            self.input_dim = output_dim * block.rate
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        self.cam = x
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.cam

# ResNet98
def resnet_student(output_dim=10):
    return ResNetStudent(BottleNeck, [6,8,12,6], num_classes=output_dim)

# ResNet50
def resnet_teacher(output_dim=10):
    return ResNetAssistant(BottleNeck, [4,6,9,4], num_classes=output_dim)
    
# モデルのパラメータ数のカウント
# teacher = resnet_teacher()
# student = resnet_student()
teacher = Teacher()
student = Student()

teacher.eval()
student.eval()

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

print(f"Teacher parameters: {count_parameters(teacher)}")
print(f"Student parameters: {count_parameters(student)}")

