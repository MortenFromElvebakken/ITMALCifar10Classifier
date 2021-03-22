import torch
from torch import nn

#Der findes indbygget vgg nets, men vi bygger vores eget og træner det
class VGGnet2ConvLayer(nn.Module):
    def __init__(self, in_size, out_size, kernels, normalize=True):
        super(VGGnet2ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_size,out_size,kernels,stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_size,out_size,kernels,stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

class VGGnet4ConvLayer(nn.Module):
    def __init__(self, in_size, out_size, kernels, normalize=True):
        super(VGGnet4ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_size,out_size,kernels,stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_size,out_size,kernels,stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernels, stride=1, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, kernels, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

class VGGnetFullyConnected(nn.Module):
    def __init__(self, in_size, out_size,final_size):
        super(VGGnetFullyConnected, self).__init__()
        self.fc1 = nn.Linear(in_size*2*2,out_size)
        self.fc2 = nn.Linear(out_size, out_size)
        self.fc3 = nn.Linear(out_size, final_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = nn.dropout(x)
        x = self.fc3(x)
        return x

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.first = VGGnet2ConvLayer(3,64,3)
        self.second = VGGnet2ConvLayer(64,128,3)
        self.third = VGGnet4ConvLayer(128,264,3)
        self.fourth = VGGnet4ConvLayer(264,512,3)
        self.fifth = VGGnetFullyConnected(512, 4096, 10)

    def forward(self, input):
        x1 = self.first(input)
        x2 = self.second(x1)
        x3 = self.third(x2)
        x4 = self.fourth(x3)
        x4 = x4.view(4, -1) #Ret til batch size tal hvis vi ændrer batch size
        x5 = self.fifth(x4)
        #output = nn.Softmax(x5)
        return x5


