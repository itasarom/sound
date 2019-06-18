
import torch
import os
import sys
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split

import librosa
from IPython import display

from torch import nn
import torch.nn.functional as F
import time

from torch.utils.data import Dataset, DataLoader

import utils



def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0, 
        scores[nonzero_weight_sample_indices, :], 
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap



class SoundDataset(Dataset):
    def __init__(self, all_data, y):
        super().__init__()
        self.all_data = all_data
        self.y = y
        
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        x = self.all_data[idx].T
        result = {
            "x":x,
            "y":self.y[idx],
            "mask":np.ones(x.shape[:-1])
        }
        
        return result



def mix_sounds(all_data, y, n, params):

    

    ids = []
    coefs = []
    for i in range(n):
        coef = np.random.rand()
        ids.append(np.random.randint(0, len(all_data) - 1))
        coefs.append(coef)

    coefs = np.array(coefs)
    coefs /= coefs.sum()
    max_len = max([len(all_data[id]) for id in ids])

    res_x = np.zeros((max_len, ))
    res_y = np.zeros_like(y[0])
    for coef, id in zip(coefs, ids):
        cur = all_data[id]
        res_x[:len(cur)] += coef * cur
        # res_y += coef * y[id]
        res_y[y[id] != 0] = 1


    return res_x, res_y, coefs, ids





class SoundAugDataset(Dataset):
    # def __init__(self, all_data, y, transform, max_size, max_n_mixed, seed):
    def __init__(self, all_data, y, transform, config, params, room_filters, seed):
        super().__init__()
        np.random.seed(seed)

        self.params = params
        self.all_data = all_data
        self.y = y

        self.config = config
        # max_duration, max_size, max_n_mixed
        self.max_size = params["max_size"]
        self.transform = transform
        self.max_n_mixed = params["max_n_mixed"]

        self.room_filters = room_filters
        
    
    def __len__(self):
        return self.max_size
    
    def __getitem__(self, idx):
        x, y, coefs, ids = mix_sounds(self.all_data, self.y, np.random.randint(1, self.max_n_mixed + 1), self.params)


        max_bits = self.config.sampling_rate * self.config.duration
        # print(max_bits, len(x))
        if len(x) > max_bits:
            start_crop = np.random.randint(0, len(x) - max_bits)
            x = x[start_crop:start_crop + max_bits]
            # print(max_bits, len(x))


        if "change_pitch" in self.params:
            if np.random.rand() < self.params["change_pitch"]:
                delta = np.random.randint(-self.params["change_pitch_max"], self.params["change_pitch_max"])
                x = librosa.effects.pitch_shift(x, sr=self.config.sampling_rate, n_steps=delta)


        if "add_echo" in self.params:
            if np.random.rand() < self.params["add_echo"]:
                #np.random.randint(len(self.room_filters)//20)
                x = utils.apply_filter(x, self.room_filters[np.random.randint(len(self.room_filters))])
        

        if "harmonic_percussive" in self.params:
            if np.random.rand() < self.params["harmonic_percussive"]:
                if np.random.randint(0, 2):
                    x = librosa.effects.percussive(x)
                else:
                    x = librosa.effects.harmonic(x)
        
        if "noise_magnitude" in self.params:
            x += self.params["noise_magnitude"] * np.random.randn(*x.shape)


        if len(x) > max_bits:
            start_crop = np.random.randint(0, len(x) - max_bits)
            x = x[start_crop:start_crop + max_bits]


        x_res = self.transform(x).T
        result = {
            "x":x_res,
            "raw_x":x,
            "y":y,
            "coefs":coefs,
            "ids":ids
        }
        
        return result
        


def collate_fn(samples):
    x = []
    y = []
    mask = []
    
    for s in samples:
        x.append(s["x"][None])
        y.append(s["y"][None])
        mask.append(s["mask"][None])
    
    
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    mask = np.concatenate(mask, axis=0)
    
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    mask = torch.Tensor(mask)
    
    return dict(x=x, y=y, mask=mask)


def collate_fn_2(samples):
    x = []
    y = []
    mask = []
    
    for s in samples:
        x.append(s["x"][None])
        y.append(s["y"][None])
        mask.append(s["mask"][None])
    
    
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    mask = np.concatenate(mask, axis=0)
    
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    mask = torch.Tensor(mask)
    
    return [x, y]



class AugmentationCollator:
    def __init__(self, config, transform, return_mode=None):
        self.config = config
        self.transform = transform
        self.return_mode = return_mode

    def __call__(self, samples):
        x = []
        y = []
        mask = []
        ss = []
        raw_x = []


        # max_len = max([len(s['x']) for s in samples])


        for s in samples:
            cur_x = s['x']
            # padding = max_len - len(cur_x)    # add padding at both ends
            # offset = padding // 2
            # if padding > 0:
                # offset = np.random.randint(0, padding)
                # cur_x = np.pad(cur_x, (offset, max_len - len(cur_x) - offset), 'constant')
            # else:
                # cur_x = 
            # else:
                # offset = 0
            
            ss.append(cur_x)

            # print(cur_x)

            # cur_x = cur_x)
            x.append(torch.Tensor(cur_x))
            # print(x[-1].shape)
            y.append(s["y"][None])
            raw_x.append(s['raw_x'])
            # print(cur_x.shape)
            # mask.append(s["mask"][None])
        
        
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
        # x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        # mask = np.concatenate(mask, axis=0)
        
        # x = torch.Tensor(x)
        y = torch.Tensor(y)
        # mask = torch.Tensor(mask)
        
        if self.return_mode is None:
            return dict(x=x, y=y, ss=ss, raw_x=raw_x)
        else:
            return [x, y]


def loss_function(logits, y):
    pred = F.log_softmax(logits, dim=1)
    res = (pred * y).sum(dim=-1)
    return -res.mean()


class Trainer:
    def __init__(self, model, optimizer_factory, scheduler_factory, device, trainer_params):
        self.model = model
        self.device = device
        self.optimizer = optimizer_factory(model.parameters())

        self.scheduler = scheduler_factory(self.optimizer)


        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.total_iterations = 0

        self.all_params = []
        self.total_epochs = 0

        self.trainer_params = trainer_params

        path = trainer_params["path"]
        new_path = os.path.join(path, time.strftime("%Y-%m-%d.%H.%M.%S", time.gmtime(time.time())))
        os.makedirs(new_path)

        self.save_path = new_path

        print("Saving the weights and metrics to ", self.save_path)




    def validate(self, loader):
        self.model.eval()
        losses = 0
        target_metrics = 0
        total_x = 0
        for b in loader:
            x = b['x'].to(self.device)
            # mask = b['mask'].to(self.device)
            y = b['y'].to(self.device)
            # logits = self.model.forward(x, mask)
            logits = self.model.forward(x)
            loss = loss_function(logits, y)
            losses += loss.item() * len(x)
            target_metric = calculate_overall_lwlrap_sklearn(y.cpu().numpy(), logits.detach().cpu().numpy())
            target_metrics += target_metric * len(x)
            total_x += len(x)
            
        losses /= total_x
        target_metrics /= total_x
        
        return {"loss":losses, "target_metric":target_metrics}


    def plot(self, train_metrics, val_metrics):
        display.clear_output()
        for key in train_metrics:
            
            plt.figure(figsize=(15, 10))
            plt.title(key)
            plt.plot(train_metrics[key], label='train_' + key)
            plt.plot(val_metrics['iterations'], val_metrics[key], label='val_' + key)
            plt.legend()
            plt.show()


    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_weights.tc"))
        torch.save((self.train_metrics, self.val_metrics), os.path.join(self.save_path, "metrics.tc"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.save_path, "optimizer.tc"))



    def train(self, loaders, params):

        train_loader = loaders["train_loader"]
        test_loader = loaders["test_loader"]

        self.all_params.append(params)

        for epoch_id in range(self.total_epochs, self.total_epochs + params['epochs']):
            self.total_epochs = epoch_id
            start_time = time.time()
            for iteration_id, batch in enumerate(train_loader, start=self.total_iterations):
                self.model.train()
                
                x = batch['x'].to(self.device)
                # mask = batch['mask'].to(self.device)
                y = batch['y'].to(self.device)
                # logits = self.model.forward(x, mask)
                # print(y.shape)
                logits = self.model.forward(x)
                loss = loss_function(logits, y)
                self.optimizer.zero_grad()
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), params["grad_clip_norm"])

                loss.backward()
                self.optimizer.step()
                
                self.train_metrics['loss'].append(loss.item())
                target_metric = calculate_overall_lwlrap_sklearn(y.cpu().numpy(), logits.detach().cpu().numpy())
                self.train_metrics['target_metric'].append(target_metric)
                    

                self.plot(self.train_metrics, self.val_metrics)
                
                if self.total_iterations % params["validate_every"]:
                    validation_result = self.validate(test_loader)
                    
                    self.val_metrics['iterations'].append(self.total_iterations)
                    for key, value in validation_result.items():
                        self.val_metrics[key].append(value)

                    self.save()
                        
                self.total_iterations = iteration_id

            validation_result = self.validate(test_loader)
            self.last_epoch_duration = time.time() - start_time
                    
            self.val_metrics['iterations'].append(self.total_iterations)
            for key, value in validation_result.items():
                self.val_metrics[key].append(value)

            self.save()
            self.scheduler.step(validation_result['loss'])


    def predict(loader):
        self.model.eval()
        result = []
        for b in loader:
            x = b['x'].to(device)
            # mask = b['mask'].to(device)
            pred = self.model.forward(x, mask)
            
            result.append(pred.detach().cpu().numpy())
            
        
        result = np.concatenate(result, axis=0)
        self.model.train()
        
        return result
        
        

            
def validate_final(model, loader, device):
        model.eval()
        losses = 0
        target_metrics = 0
        total_x = 0
        loss_function = nn.BCEWithLogitsLoss()
        for b in loader:
            x = b['x'].to(device)
            # mask = b['mask'].to(self.device)
            y = b['y'].to(device)
            # logits = self.model.forward(x, mask)
            logits = model.forward(x)
            loss = loss_function(logits, y)
            losses += loss.item() * len(x)
            target_metric = calculate_overall_lwlrap_sklearn(y.cpu().numpy(), logits.detach().cpu().numpy())
            target_metrics += target_metric * len(x)
            total_x += len(x)
            
        losses /= total_x
        target_metrics /= total_x
        
        return {"loss":losses, "target_metric":target_metrics}
    
