def conget_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(dataset_name))
    return globals()[dataset_name]
    
class supervised():
    def __init__(self):
        super(supervised, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'learning_rate': 1e-3,
        }
        self.alg_hparams_edf = {
            'dsn': {'features_len': 41 * 128, 'clf':1024},
            'attnsleep': {'features_len': 30 * 80, 'clf': 2400},
            'cnn1d': {'features_len': 65 * 128, 'clf':65 * 128}
        }
        self.alg_hparams_shhs = {
            'dsn': {'features_len': 51 * 128, 'clf': 1024},
            'attnsleep': {'features_len': 99 * 30, 'clf':2970},
            'cnn1d': {'features_len': 80 * 128, 'clf':80 * 128}
        }
        self.alg_hparams_isruc = {
            'dsn': {'features_len': 80 * 128, 'clf': 1024},
            'attnsleep': {'features_len': 157 * 30, 'clf':157 * 30},
            'cnn1d': {'features_len': 128 * 128, 'clf':128 * 128}
        }


class clsTran():
    def __init__(self):
        super(clsTran, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'learning_rate': 1e-3,
        }
        self.alg_hparams_edf = {
            'dsn': {'features_len': 41 * 128, 'clf': 1024},
            'attnsleep': {'features_len': 30 * 80, 'clf':2400},
            'cnn1d': {'features_len': 65 * 128, 'clf':65 * 128}
        }
        self.alg_hparams_shhs = {
            'dsn': {'features_len': 51 * 128, 'clf': 1024},
            'attnsleep': {'features_len': 99 * 30, 'clf':99 * 30},
            'cnn1d': {'features_len': 80 * 128, 'clf':80 * 128}
        }
        self.alg_hparams_isruc = {
            'dsn': {'features_len': 80 * 128, 'clf': 1024},
            'attnsleep': {'features_len': 157 * 30, 'clf':157 * 30},
            'cnn1d': {'features_len': 128 * 128, 'clf':128 * 128}
        }
