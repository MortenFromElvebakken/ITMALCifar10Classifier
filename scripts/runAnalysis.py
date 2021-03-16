import logging
import click
import sys
import os

from src.dataLayer import analyzeCifar10


@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """


    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()