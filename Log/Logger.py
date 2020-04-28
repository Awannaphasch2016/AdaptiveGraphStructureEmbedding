# coding: utf-8

import logging


class Logger(object):

    def __init__(self, name='logger', level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        fh = logging.FileHandler('%s.log' % name, 'w')
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        self.logger.addHandler(sh)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


if __name__ == '__main__':
    log = Logger()
    log.debug('debug')
    log.info('info')
    log.warning('warning')
    log.error('error')