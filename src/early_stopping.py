import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=2, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 2
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        
        loss = val_loss

        if self.best_loss is None:
            self.best_loss = loss
            # self.save_checkpoint(val_loss, model)
        elif loss > self.best_loss:
            
            if loss < self.best_loss + self.delta:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}, best_loss is {self.best_loss}, in tolerance{self.delta}')
                pass
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}, best_loss is {self.best_loss}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_loss = loss
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     path = os.path.join(self.save_path, 'best_network.pth')
    #     torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
    #     self.val_loss_min = val_loss