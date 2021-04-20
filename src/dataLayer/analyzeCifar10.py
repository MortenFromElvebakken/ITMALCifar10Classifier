import torch
import torchvision.transforms as transforms
from src.dataLayer.CreateDataloaders import CreateDataloaders
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#inspireret fra https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?fbclid=IwAR3UzylpXP1ob0MZd-Ic3BZZKfs0zIcgqaxGl6qtjqw6M3F05V1ufpmW5j8
class AnalyzeCifar10():
    def __init__(self):
        self.dataloaderobj = CreateDataloaders(normalize=False)
        self.trainLoader, self.testLoader, classes = self.dataloaderobj.getDataloaders()

    def printHistogram(self,index,name):

        #For individual class
        #data = self.dataloaderobj.getDataset(index)

        #For full dataset
        data = self.trainLoader.dataset.data
        df = []

        for i in data:
            r, g, b = cv2.split(i)
            b = b.flatten()
            r = r.flatten()
            g = g.flatten()
            df.append(pd.DataFrame(np.stack([r, g, b], axis=1), columns=['Red', 'Green', 'Blue']))
        d = {'color': ['r', 'g', 'b']}
        df_merged = pd.concat(df)
        self.getMedian(df_merged,name)
        self.getMean(df_merged, name)
        self.getstd(df_merged, name)
        axes = df_merged.plot(kind='hist', subplots=True, layout=(3, 1), bins=256, color=['r', 'g', 'b'], yticks=[],
                              sharey=True, sharex=True)
        axes[0, 0].yaxis.set_visible(False)
        axes[1, 0].yaxis.set_visible(False)
        axes[2, 0].yaxis.set_visible(False)
        fig = axes[0, 0].figure
        fig.text(0.5, 0.04, "Pixel Value", ha="center", va="center")
        fig.text(0.05, 0.5, "Pixel frequency", ha="center", va="center", rotation=90)
        # plt.xlim(0, 4000)

        plt.title(name)
        plt.show()

    def getMedian(self,df,name):
        Red_median = df['Red'].median()
        Gren_median = df['Green'].median()
        Blue_median = df['Blue'].median()
        print("Red median for "+str(name)+" is "+str(round(Red_median,2)))
        print("Green median for " + str(name) + " is " + str(round(Gren_median,2)))
        print("Blue median for " + str(name) + " is " + str(round(Blue_median,2)))
    def getMean(self,df,name):
        Red_median = df['Red'].mean()
        Gren_median = df['Green'].mean()
        Blue_median = df['Blue'].mean()
        print("Red mean for " + str(name) + " is " + str(round(Red_median,2)))
        print("Green mean for " + str(name) + " is " + str(round(Gren_median,2)))
        print("Blue mean for " + str(name) + " is " + str(round(Blue_median,2)))
    def getstd(self,df,name):
        Red_median = df['Red'].std()
        Gren_median = df['Green'].std()
        Blue_median = df['Blue'].std()
        print("Red std for " + str(name) + " is " + str(round(Red_median,2)))
        print("Green std for " + str(name) + " is " + str(round(Gren_median,2)))
        print("Blue std for " + str(name) + " is " + str(round(Blue_median,2)))


    def showImage(self,image):
        #img = image / 2 + 0.5  # bliver nødt til at unnormalize, siden vi har det i tensor fra pytorch
        npimg = image.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


#vise et billede fra hver klasse
#Et histogram af pixel fordeling for hele datasæt og et for hver klasse
#Midellværdi for hver klasse og hele pd.var()

