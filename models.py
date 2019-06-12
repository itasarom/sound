
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class AttentionMap(nn.Module):
    """ A Layer which provides attention map for a query. """
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_query = nn.Conv1d(in_channels=query_dim, out_channels=hidden_dim,kernel_size=1, bias=False)
        self.proj_key = nn.Conv1d(in_channels=key_dim, out_channels=hidden_dim, kernel_size=1, bias=False)
    
    def forward(self, query, key, mask=None):
        """
        query: B, T, D1
        key: B, T, D2
        mask: B, T
        -> B, T, T
        
        
        """
        
        query = self.proj_query(query.transpose(1, 2)).transpose(1, 2)
        key = self.proj_key(key.transpose(1, 2)).transpose(1, 2)
        
        weights = torch.einsum("btd,bed->bte",(query, key))
        weights /= self.hidden_dim ** 0.5
        weights = F.softmax(weights, dim=2)
        if mask is not None:
            weights = weights * mask.unsqueeze(dim=1)
            weights /= weights.sum(dim=2, keepdim=True)
        
        
        return weights


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_attention_dim, dropout):
        super().__init__()
        self.dropout = dropout
#         self.max_length = max_length
        self.attention_map = AttentionMap(query_dim, key_dim, hidden_attention_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=value_dim)
    
    
    def forward(self, query, key, value, mask=None):
        
        
        attention_weights = self.attention_map(query, key, mask)
        output = (value.unsqueeze(1) + value.unsqueeze(1) * attention_weights.unsqueeze(-1)).sum(dim=2)
        output = self.layer_norm(output)
        output = F.dropout(output, self.dropout)
        
        return output
    

class SelfAttention(nn.Module):
    def __init__(self, n_dims, hidden_attention_dim, dropout):
        super().__init__()
        self.n_dims = n_dims
        self.attention = Attention(n_dims, n_dims, n_dims, hidden_attention_dim, dropout)
        
    
    def forward(self, value, mask=None):
        result = self.attention(value, value, value, mask)
        
        return result


class TimeLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.layer = torch.nn.Linear(in_features, out_features, bias)
        
    
    def forward(self, input):
        result = torch.einsum("btd,od->bto", (input, self.layer.weight)) 
        if self.layer.bias is not None:
            result += self.layer.bias.unsqueeze(0).unsqueeze(0)
            
        return result
    

class TransformerBlock(nn.Module):
    def __init__(self, input_dims, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(input_dims, input_dims, dropout=dropout)
        self.time_linear = TimeLinear(input_dims, input_dims)
        self.activation = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(input_dims)
        
    def forward(self, input):
        x = self.attention(input)
        y = self.time_linear(x)
        y = self.activation(y)
        y += x
        
        result = self.layer_norm(y)
        
        return y
        

class Net(nn.Module):
    def __init__(self, input_dims, max_length, n_classes):
        super().__init__()
        self.input_dims = input_dims
        self.max_length = max_length
        self.n_classes = n_classes
        
        
        self.attention = nn.Sequential(
               TransformerBlock(128),
              TransformerBlock(128),
                TransformerBlock(128),
        )
        

        self.classifier_net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.input_dims * self.max_length, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.n_classes)
        )
        
    
    # def forward(self, input, mask):
    def forward(self, input):
        batch_size = input.shape[0]
        
        x = self.attention(input)
        x = x.reshape(batch_size, -1)
        
        classes = self.classifier_net(x)
        
        return classes



class LSTMModel(nn.Module):
    def __init__(self, input_dims,  n_classes):
        super().__init__()
        self.input_dims = input_dims
        self.n_classes = n_classes

        self.num_layers = 3
        self.num_dir = 2
        self.hidden_dim = 128
        
        
        self.lstm = nn.GRU(self.input_dims, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=(self.num_dir==2), dropout=0.1)
        

        self.classifier_net = nn.Sequential(
            nn.Linear(self.num_dir * self.hidden_dim, 1024),
            nn.LeakyReLU(),
            # nn.Linear(1024, 256),
            # nn.LeakyReLU(),
            nn.Linear(1024, self.n_classes)
        )
        
    
    def forward(self, input, mask):
        batch_size = input.shape[0]
        
        output, _ = self.lstm(input)
        
        # h = h.view(self.num_layers, self.num_dir, batch_size, -1)
        # h = h.permute(2, 1, 0, 3)
        # h = h.reshape(batch_size, -1)
        output = output[:, -1]
        
        classes = self.classifier_net(output)
        
        return classes