import torch
import torch.nn as nn



class GRUTransition(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, activation:str) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "elu":
            activation_fn = nn.ELU()
        else:
            raise ValueError("Unknown activation function")
        
        self.linear_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation_fn
        )

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    
    def initialize_hidden_state(self, batch_size:int) -> torch.Tensor:
        '''
        initialize the hidden state for GRU forward, shape (B, hidden_dim)
        '''

        # Create the hidden state on the model's device.
        device = next(self.parameters()).device

        h_0 = torch.zeros(batch_size, self.hidden_dim, device=device)

        return h_0
    
    def forward(self, s_last_t, a_last_t, h_last_t):
        '''
        perform 1 recurrence step from t-1 to t
        '''

        # Project concatenated state and action to the recurrent hidden size.
        proj = torch.cat((s_last_t, a_last_t), dim=-1)
        proj = self.linear_proj(proj)

        h_new = self.gru(proj, h_last_t)

        return h_new
