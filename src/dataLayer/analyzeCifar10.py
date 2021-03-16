import torch
from src.dataLayer.CreateDataloaders import CreateDataloaders
import matplotlib as plt
import numpy as np
import seaborn as sns

#inspireret fra https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?fbclid=IwAR3UzylpXP1ob0MZd-Ic3BZZKfs0zIcgqaxGl6qtjqw6M3F05V1ufpmW5j8
class AnalyzeCifar10():
    def __init__(self):
        test = CreateDataloaders(normalize=False)
        trainLoader, testLoader, classes = test.getDataloaders()


    def showImage(self,image):
        #img = image / 2 + 0.5  # bliver nødt til at unnormalize, siden vi har det i tensor fra pytorch
        npimg = image.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()



