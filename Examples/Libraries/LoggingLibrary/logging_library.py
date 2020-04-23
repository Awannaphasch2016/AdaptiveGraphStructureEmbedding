import logging
import os.path as osp
path = osp.join(osp.dirname(osp.realpath(__file__)))
logging.basicConfig(filename=f'{path}/example.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
