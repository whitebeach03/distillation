import torch
import torch.nn as nn

class TeacherConvBnAct(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class StudentConvBnAct(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def _make_teacher_layer(self, n, input_dim, output_dim):
        layers = []
        for _ in range(n):
            layers.append(TeacherConvBnAct(input_dim, output_dim))
            input_dim = output_dim
        return nn.Sequential(*layers)
    
    def _make_student_layer(self, n, input_dim, output_dim):
        layers = []
        for _ in range(n):
            layers.append(StudentConvBnAct(input_dim, output_dim))
            input_dim = output_dim
        return nn.Sequential(*layers)

# 16-layer model
class StudentModel(BaseModel):
    def __init__(self, input_dim=3):
        super().__init__()
        self.layer1 = self._make_student_layer(4, input_dim, 16)
        self.layer2 = self._make_student_layer(4, 16, 32)
        self.layer3 = self._make_student_layer(4, 32, 64)
        self.layer4 = self._make_student_layer(4, 64, 128)
        self.mlp = MLP(128, 128, 10)
        self.maxpool = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x
    
# 12-layer model
class TeacherAssistantModel(BaseModel):
    def __init__(self, input_dim=3):
        super().__init__()
        self.layer1 = self._make_student_layer(3, input_dim, 32)
        self.layer2 = self._make_student_layer(3, 32, 64)
        self.layer3 = self._make_student_layer(3, 64, 128)
        self.layer4 = self._make_student_layer(3, 128, 256)
        self.mlp = MLP(256, 128, 10)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x
        
# 8-layer model
class TeacherModel(BaseModel):
    def __init__(self, input_dim=3):
        super().__init__()
        self.layer1 = self._make_teacher_layer(2, input_dim, 64)
        self.layer2 = self._make_teacher_layer(2, 64, 128)
        self.layer3 = self._make_teacher_layer(2, 128, 256)
        self.layer4 = self._make_teacher_layer(2, 256, 512)
        self.mlp = MLP(512, 128, 10)
        self.maxpool = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x

class Model(BaseModel):
    def __init__(self, input_dim=3):
        super().__init__()
        self.layer1 = self._make_student_layer(1, input_dim, 8)
        self.layer2 = self._make_student_layer(1, 8, 16)
        self.layer3 = self._make_student_layer(1, 16, 32)
        self.layer4 = self._make_student_layer(1, 32, 64)
        self.mlp = MLP(256, 128, 4)
        self.maxpool = nn.MaxPool2d(3, stride=3)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x

class TModel(BaseModel):
    def __init__(self, input_dim=3):
        super().__init__()
        self.layer1 = self._make_student_layer(3, input_dim, 32)
        self.layer2 = self._make_student_layer(3, 32, 64)
        self.layer3 = self._make_student_layer(3, 64, 128)
        self.layer4 = self._make_student_layer(3, 128, 256)
        self.mlp = MLP(1024, 128, 4)
        self.maxpool = nn.MaxPool2d(3, stride=3)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x
    
    
### ResNet ###
class BasicBlock(nn.Module):
    rate=1
    def __init__(self, input_dim, output_dim,stride=1,drop=0.3):
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
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
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

class SampleModel(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        self.attmap = x
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [self.attmap, self.fc.weight]
        

# ResNet98
def resnet_student(output_dim=10):
    return ResNetStudent(BottleNeck, [6,8,12,6], num_classes=output_dim)

# ResNet50
def resnet_teacher(output_dim=10):
    return ResNetAssistant(BottleNeck, [4,6,9,4], num_classes=output_dim)
    
# モデルのパラメータ数のカウント
teacher = resnet_teacher()
student = resnet_student()

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

print(f"Teacher parameters: {count_parameters(teacher)}")
print(f"Student parameters: {count_parameters(student)}")