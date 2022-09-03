import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init

'''
    summary(model=self, input_size=[(27,348), (27,64)], batch_size=666)
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                  [666, 64]          22,336
                GRUCell-2                 [666, 64]               0
                Linear-3                  [666, 36]           2,340
    ================================================================
    Total params: 24,676
    Trainable params: 24,676
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 41249.72
    Forward/backward pass size (MB): 0.83
    Params size (MB): 0.09
    Estimated Total Size (MB): 41250.65
    ----------------------------------------------------------------
'''

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(NRNNAgent, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)
        
        # self.apply(weights_init)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new_zeros(1, self.rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q.view(b, a, -1), h.view(b, a, -1)

    # def forward_new(self, inputs, hidden_state):

