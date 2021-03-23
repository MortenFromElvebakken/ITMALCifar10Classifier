import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class trainInpainting():
    def __init__(self, trainingImages, vggNet, path):
        self.training = trainingImages
        self.vggNet = vggNet
        self.epochs = 1
        self.path = path

    def traingan(self):

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.vggNet.parameters(), lr=0.001, momentum=0.9) #torch.optim.Adam(self.vggNet.parameters(), lr=0.001, betas=(0.9,0.99))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        def weights_init(m):
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


        self.vggNet = self.vggNet.apply(weights_init)
        self.vggNet.to(device)
        train_loss = 0.0
        i = 1
        for epoch in range(self.epochs):
            # Dataloader returns the batches
            RunningLoss = 0.0
            for batchOfImages, labels in tqdm(self.training):

                batchOfImages = batchOfImages.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = self.vggNet(batchOfImages)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if i % 2500 == 2499:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, train_loss / 2500))
                    train_loss = 0.0
                i = i+1
                #break
        #torch.save(self.vggNet.state_dict(), self.path)
        return self.vggNet