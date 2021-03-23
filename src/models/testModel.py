import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class testInpainting():
    def __init__(self, testImages, vggNet, classes):
        self.vggNet = vggNet
        self.testImages = testImages
        self.classes = classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def runTest(self):
        TotalGuesses = 0
        CorrectGuesses = 0

        WithinClassCorrect = list(0. for i in range(10))
        WithinClassTotal = list(0. for i in range(10))

        with torch.no_grad():
            for batchOfImages, labels in tqdm(self.testImages):
                batchOfImages = batchOfImages.to(self.device)
                labels = labels.to(self.device)
                outputs = self.vggNet(batchOfImages)
                _, PredictedLabels = torch.max(outputs.data, 1)
                TotalGuesses += labels.size(0)
                CorrectGuesses += (PredictedLabels == labels).sum().item()

                c = (PredictedLabels == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    WithinClassCorrect[label] += c[i].item()
                    WithinClassTotal[label] += 1
        print('Score of the classifier on the 10000 test images: %d %%' % (100 * CorrectGuesses / TotalGuesses))
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * WithinClassCorrect[i] / WithinClassTotal[i]))