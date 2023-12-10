import torch
import os
import random
import numpy as np


class EarlyStopping:
    def __init__(self, patience, verbose):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
    
    def __call__(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss
        return False
    
def save_param(es, loss, model, model_type, seed):
    if es(loss):
        return True
    else:
        torch.save(model.state_dict(), './logs/resnet/' + model_type + '/' + str(seed) + '_param.pth')
        return False 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True