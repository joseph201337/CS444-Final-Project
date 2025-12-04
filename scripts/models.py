import torch
import torch.nn as nn

class RainANN(nn.Module):
    def __init__(self, input_dim, seq_length):
        super(RainANN, self).__init__()
        self.flat_dim = input_dim * seq_length 
        
        self.layer1 = nn.Linear(self.flat_dim, 128)
        
        self.act1 = nn.LeakyReLU() 
        
        self.dropout = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(128, 64)
        
        self.act2 = nn.LeakyReLU()
        
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        
        x = self.act1(self.layer1(x))
        x = self.dropout(x)
        
        x = self.act2(self.layer2(x))
        
        x = self.output(x)
        return x

class RainLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(RainLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.5,
        )
        self.fc = nn.Linear(hidden_dim , 1)  # Predict from last hidden state
        
    def forward(self, x):
        # x shape: (Batch, Seq, Feat)
        out, (h_n, c_n) = self.lstm(x)
        # Take the output of the LAST time step
        # out[:, -1, :] shape is (Batch, Hidden_Dim)
        last_time_step = out[:, -1, :] 
        prediction = self.fc(last_time_step)
        return prediction


class RainGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(RainGRU, self).__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.4
        )
        self.fc = nn.Linear(hidden_dim, 1)  # Predict from last hidden state
        
    def forward(self, x):
        # x shape: (Batch, Seq, Feat)
        # GRU returns: output, hidden_state 
        out, h_n = self.gru(x)
        # Take the output of the LAST time step
        # out[:, -1, :] shape is (Batch, Hidden_Dim)
        last_time_step = out[:, -1, :] 
        prediction = self.fc(last_time_step)
        return prediction