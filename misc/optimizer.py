import numpy as np
from torch import optim

def get_optimizer(config, model):
    optimizer = base_optimizer(config, model)
    return optimizer

class base_optimizer(object):
    def __init__(self, config, model):
        super(base_optimizer, self).__init__()

        # optimizer
        if config.optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
        else:
            raise

        # scheduler
        if config.scheduler == 'ReduceLROnPlateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        elif config.scheduler == 'StepLR':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(self.optimizer, 100, gamma=0.1, last_epoch=-1)
        else:
            self.scheduler = None

    def update_model(self, loss):
        self.optimizer.zero_grad()

        if np.isnan(loss.item()):
            print('\n\n\nERROR::: THE LOSS IS NAN\n\n\n')
            raise()
        else:
            loss.backward()
            self.optimizer.step()
        return None

    def scheduler_step(self, metric):
        if self.scheduler is not None:
            self.scheduler.step(metric)
        return None