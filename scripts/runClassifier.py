import logging
import click
import sys
import os
from src.models.trainModel import trainInpainting
from src.models.convNet import Vgg19
from src.dataLayer.CreateDataloaders import CreateDataloaders


@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    vggNet = Vgg19()

    DatLayer = CreateDataloaders()
    trainloader, testLoader, classes = DatLayer.getDataloaders()
    trainingClass = trainInpainting(trainloader, testLoader,vggNet)
    trainingClass.traingan()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()