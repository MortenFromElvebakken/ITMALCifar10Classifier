import logging
import click
import sys
import os
from src.models.trainModel import trainInpainting
from src.models.testModel import testInpainting
from src.models.convNet import Vgg19
from src.dataLayer.CreateDataloaders import CreateDataloaders


@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    #Set logger
    logger = logging.getLogger(__name__)

    #Create model
    vggNet = Vgg19()

    #Datalayer
    DatLayer = CreateDataloaders()
    trainloader, testLoader, classes = DatLayer.getDataloaders()
    path = r'C:\Users\Morten From\PycharmProjects\ITMALClassifierCifar10\src\models'

    #Training
    trainingClass = trainInpainting(trainloader,vggNet, path)
    vggNet = trainingClass.traingan()

    #Testing
    testClass = testInpainting(testLoader, vggNet, classes)
    testClass.runTest()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()