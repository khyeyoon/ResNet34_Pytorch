import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from utils import progress_bar
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

class Resnet(nn.Module):
  def __init__(self, num_classes = 100):
    super(Resnet, self).__init__()
    
    #conv1
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    
    #conv2
    self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv2_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.bn4 = nn.BatchNorm2d(64)
    self.conv2_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.bn5 = nn.BatchNorm2d(64)
    self.conv2_6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.bn6 = nn.BatchNorm2d(64)
    self.conv2_7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.bn7 = nn.BatchNorm2d(64)
    
    #conv3
    self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2,padding=1)
    self.bn8 = nn.BatchNorm2d(128)
    self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.bn9 = nn.BatchNorm2d(128)
    self.conv3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.bn10 = nn.BatchNorm2d(128)
    self.conv3_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.bn11 = nn.BatchNorm2d(128)
    self.conv3_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.bn12 = nn.BatchNorm2d(128)
    self.conv3_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.bn13 = nn.BatchNorm2d(128)
    self.conv3_7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.bn14 = nn.BatchNorm2d(128)
    self.conv3_8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.bn15 = nn.BatchNorm2d(128)
    
    #conv4
    self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2,padding=1)
    self.bn16 = nn.BatchNorm2d(256)
    self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn17 = nn.BatchNorm2d(256)
    self.conv4_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn18 = nn.BatchNorm2d(256)
    self.conv4_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn19 = nn.BatchNorm2d(256)
    self.conv4_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn20 = nn.BatchNorm2d(256)
    self.conv4_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn21 = nn.BatchNorm2d(256)
    self.conv4_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn22 = nn.BatchNorm2d(256)
    self.conv4_8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn23 = nn.BatchNorm2d(256)
    self.conv4_9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn24 = nn.BatchNorm2d(256)
    self.conv4_10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn25 = nn.BatchNorm2d(256)
    self.conv4_11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn26 = nn.BatchNorm2d(256)
    self.conv4_12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.bn27 = nn.BatchNorm2d(256)
    
    #conv5
    self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2,padding=1)
    self.bn28 = nn.BatchNorm2d(512)
    self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1)
    self.bn29 = nn.BatchNorm2d(512)
    self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1)
    self.bn30 = nn.BatchNorm2d(512)
    self.conv5_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1)
    self.bn31 = nn.BatchNorm2d(512)
    self.conv5_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1)
    self.bn32 = nn.BatchNorm2d(512)
    self.conv5_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1)
    self.bn33 = nn.BatchNorm2d(512)
    
    #batch normalization
    self.bn_s2 = nn.BatchNorm2d(128)
    self.bn_s3 = nn.BatchNorm2d(256)
    self.bn_s4 = nn.BatchNorm2d(512)
   
    self.avgpool = nn.AvgPool2d(kernel_size=4)
  
    self.fc = nn.Linear(512,num_classes)
    
    #shortcut dimension 조정
    self.conv2_1x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2,padding=0)
    self.conv3_1x1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2,padding=0)
    self.conv4_1x1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2,padding=0)
    self.conv5_1x1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2,padding=0)
   


  def forward(self, x):
    #conv1
    x = F.relu(self.bn1(self.conv1(x)))
    x_residual = x

    #conv2
    x = F.relu(self.bn2(self.conv2_1(x)))
    x = F.relu(self.bn3(self.conv2_2(x)))
    x_residual_2 = x

    x = F.relu(self.bn4(self.conv2_3(x+x_residual)))
    x = F.relu(self.bn5(self.conv2_4(x)))
    x_residual = x

    x = F.relu(self.bn6(self.conv2_5(x+x_residual_2)))
    x = F.relu(self.bn7(self.conv2_6(x)))
    x_residual_2 = x

    #conv3
    x = F.relu(self.bn8(self.conv3_1(x+x_residual)))
    x = F.relu(self.bn9(self.conv3_2(x)))
    x_residual = x

    #dimension 변경
    x_residual_2 =  F.relu(self.bn_s2(self.conv3_1x1(x_residual_2))) 
    x = F.relu(self.bn10(self.conv3_3(x+x_residual_2)))
    x = F.relu(self.bn11(self.conv3_4(x)))
    x_residual_2 = x

    x = F.relu(self.bn12(self.conv3_5(x+x_residual)))
    x = F.relu(self.bn13(self.conv3_6(x)))
    x_residual = x

    x = F.relu(self.bn14(self.conv3_7(x+x_residual_2)))
    x = F.relu(self.bn15(self.conv3_8(x)))
    x_residual_2 = x

    #conv4
    x = F.relu(self.bn16(self.conv4_1(x+x_residual)))
    x = F.relu(self.bn17(self.conv4_2(x)))
    x_residual = x

    #dimension 변경
    x_residual_2 = F.relu(self.bn_s3(self.conv4_1x1(x_residual_2))) 
    x = F.relu(self.bn18(self.conv4_3(x+x_residual_2)))
    x = F.relu(self.bn19(self.conv4_4(x)))
    x_residual_2 = x

    x = F.relu(self.bn20(self.conv4_5(x+x_residual)))
    x = F.relu(self.bn21(self.conv4_6(x)))
    x_residual = x

    x = F.relu(self.bn22(self.conv4_7(x+x_residual_2)))
    x = F.relu(self.bn23(self.conv4_8(x)))
    x_residual_2 = x

    x = F.relu(self.bn24(self.conv4_9(x+x_residual)))
    x = F.relu(self.bn25(self.conv4_10(x)))
    x_residual = x

    x = F.relu(self.bn26(self.conv4_11(x+x_residual_2)))
    x = F.relu(self.bn27(self.conv4_12(x)))
    x_residual_2 = x

    #conv5
    x = F.relu(self.bn28(self.conv5_1(x+x_residual)))
    x = F.relu(self.bn29(self.conv5_2(x)))
    x_residual = x
    
    #dimension 변경
    x_residual_2 = F.relu(self.bn_s4(self.conv5_1x1(x_residual_2))) 
    x = F.relu(self.bn30(self.conv5_3(x+x_residual_2)))
    x = F.relu(self.bn31(self.conv5_4(x)))
    x_residual_2 = x

    x = F.relu(self.bn32(self.conv5_5(x+x_residual)))
    x = F.relu(self.bn33(self.conv5_6(x)))

    #avg pool
    x = self.avgpool(x+x_residual_2)

    #fc layer
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    #transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = Resnet()
print(net)
net = net.to(device)
    
start_epoch = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print('LR: {:0.6f}'.format(optimizer.param_groups[0]['lr']))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
   


def adjust_learning_rate(optimizer, epoch):

    if epoch < 150:
            lr = args.lr   #0.1
            
    elif epoch < 225:
            lr = args.lr * 0.1  #0.01
            
    else:
            lr = args.lr * 0.01   #0.001     
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
      
        
for epoch in range(start_epoch, start_epoch+300):
    adjust_learning_rate(optimizer, epoch)
    
    train(epoch)
    test(epoch)