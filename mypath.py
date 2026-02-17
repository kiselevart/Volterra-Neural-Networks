import os

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            root_dir = os.path.join(_BASE, 'ucf101')
            output_dir = os.path.join(_BASE, 'ucf101_pre')
            return root_dir, output_dir
        elif database == 'hmdb51':
            root_dir = os.path.join(_BASE, 'hmdb51')
            output_dir = os.path.join(_BASE, 'hmdb51_pre')
            return root_dir, output_dir
        elif database == 'ucf10':
            root_dir = os.path.join(_BASE, 'ucf10')
            output_dir = os.path.join(_BASE, 'ucf10_pre')
            return root_dir, output_dir
        else:
            raise NotImplementedError(f'Database {database} not available.')

    @staticmethod
    def model_dir():
        return os.path.join(_BASE, 'pretrained', 'c3d-pretrained.pth')