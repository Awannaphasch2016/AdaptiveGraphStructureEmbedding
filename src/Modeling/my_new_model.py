import numpy as np
class MyNewModel:
    def __init__(self, run_gcn=None):
        assert isinstance(run_gcn, bool), f'run_gcn must have type = bool '

        self.run_gcn = run_gcn
        self.loss_per_epoch = {}

        self.total_loss = {}
        self.total_accs_dict = {}
        self.total_aucs_dict = {}

        self.aucs_hist_dict = {}
        self.accs_hist_dict = {}

    def check_input_requirement(self, dataset_dict, model_parameters_dict, boolean_dict):
        pass
    def init_model(self):
        pass
    def run_model(self):
        pass
    def run_model_once(self):
        pass
    def prepare_gan_data(self):
        pass
    def prepare_data_ind(self):
        pass

    def get_total_accs_and_aucs(self):
        for i, j in self.accs_hist_dict.items():
            x = np.array(j)
            if i not in self.total_accs_dict:
                self.total_accs_dict[i] = x
            else:
                self.total_accs_dict[i] += x

        for i, j in self.aucs_hist_dict.items():
            x = np.array(j)
            if i not in self.total_aucs_dict:
                self.total_aucs_dict[i] = x
            else:
                self.total_aucs_dict[i] += x
    def get_total_loss(self):
        # TODO where did i assign loss_per_epoch?
        # why is self.loss_per_epoch.train_loss = [None, None ,...]
        if 'train_loss' not in self.total_loss:
            self.total_loss['train_loss'] = np.array(
                self.loss_per_epoch['train_loss'])
        else:
            self.total_loss['train_loss'] += self.loss_per_epoch[
                'train_loss']
        if 'test_loss' not in self.total_loss:
            self.total_loss['test_loss'] = np.array(
                self.loss_per_epoch['test_loss'])
        else:
            self.total_loss['test_loss'] += self.loss_per_epoch[
                'test_loss']

    def create_file_naming_convension(self,
                                      naming_convention_dict,
                                      title=None,
                                      emb=None,
                                      is_train=None,
                                      scan=None,
                                      train_test=None,
                                      report=None,
                                      gan_performance=None):

        report_name = ''
        emb_name = ''
        scan_name = ''
        train_test_name = ''
        is_train_name = ''
        gan_performance_name = ''
        if title is not None:
            assert emb is None, ''
            assert scan is None, ''
            assert train_test is None, ''
            assert report is None, ''
            assert gan_performance is None, ''
        elif report is not None:
            assert is_train is not None, ''
            is_train_name = 'train_' if is_train else 'test_'
            assert emb is None, ''
            assert scan is None, ''
            assert train_test is None, ''
            assert gan_performance is None, ''
            assert title is None, ''
        elif emb is not None:
            emb_name = 'emb_'
            assert report is None, ''
            assert scan is None, ''
            assert train_test is None, ''
            assert gan_performance is None, ''
            assert title is None, ''
        elif scan is not None:
            scan_name = 'scan_'
            assert emb is None, ''
            assert report is None, ''
            assert train_test is None, ''
            assert gan_performance is None, ''
            assert title is None, ''
        elif train_test is not None:
            train_test_name = 'train_test_'
            assert emb is None, ''
            assert scan is None, ''
            assert report is None, ''
            assert gan_performance is None, ''
            assert title is None, ''
        elif gan_performance is not None:
            gan_performance_name = 'gan_performance_'
            assert emb is None, ''
            assert scan is None, ''
            assert report is None, ''
            assert train_test is None, ''
            assert title is None, ''
        else:
            raise ValueError('')

        assert isinstance(naming_convention_dict, dict), ""
        # assert time_stamp is not None, "time_stamp must be specified to avoid ambiguity"

        if naming_convention_dict['model_name'] == 'train_model':
            model_name = 'train_model_'
        elif naming_convention_dict['model_name'] == 'run_gcn':
            model_name = 'run_gcn_'
        else:
            raise ValueError('')

        if naming_convention_dict['is_downsampled']:
            is_downsampled = 'downsampled_'
        else:
            is_downsampled = ''

        label = emb_name + report_name + scan_name + train_test_name + is_train_name + gan_performance_name

        if self.run_gcn:
            file_name = f"{naming_convention_dict['time_stamp']}_{model_name}_edge_percent={naming_convention_dict['preserved_edges_percent']}_ep={naming_convention_dict['main_epoch']}_{is_downsampled}{label}"
        else:
            file_name = f"{naming_convention_dict['time_stamp']}_{model_name}_edge_percent={naming_convention_dict['preserved_edges_percent']}_ep={naming_convention_dict['main_epoch']}_gan_ep={naming_convention_dict['gan_epoch']}_{is_downsampled}{label}"

        return file_name

