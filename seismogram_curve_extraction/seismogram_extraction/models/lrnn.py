import numpy as np
import torch
import torch.nn as nn

class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=-1)
        hidden = self.rnn(combined)
        return hidden
    
    def process_sequence(self, sequence):
        hidden = torch.zeros(1, self.hidden_size)
        results = []
        for x in sequence:
            hidden = self.forward(x, hidden)
            results.append(hidden.detach().numpy())
        return np.array(results)