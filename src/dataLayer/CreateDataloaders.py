import torch
import torchvision
import torchvision.transforms as transforms

#inspireret fra https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?fbclid=IwAR3UzylpXP1ob0MZd-Ic3BZZKfs0zIcgqaxGl6qtjqw6M3F05V1ufpmW5j8
class CreateDataloaders():
    def __init__(self, normalize=True, batch_size=4, num_workers=2):
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.normalize:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose(
                [transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)

        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def getDataset(self,label):
        dataset  = torchvision.datasets.CIFAR10(root='./data',
                                                download=True)
        data_list = []
        for i in range(len(dataset)):
            if dataset.targets[i]==label:
                data_list.append(dataset.data[i])
        return data_list
    def getDataloaders(self):
        return self.trainloader, self.testloader, self.classes
