from torch import nn
import torch.nn.functional as F
import torch


class FullyConnected(nn.Module):
    def __init__(self, dim, architecture='M1'):
        super().__init__()

        if not architecture in ['M4', 'M1', 'M2', 'M3']:
            self.f = nn.Sequential(
                nn.Linear(dim[0], 60),
                nn.ReLU(),
                nn.Linear(60, 25),
                nn.ReLU(),
                nn.Linear(25, 1),
            )
        elif architecture == 'M1':
            self.f = nn.Sequential(
                nn.Linear(dim[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        elif architecture == 'M2':
            self.f = nn.Sequential(
                nn.Linear(dim[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        elif architecture == 'M3':
            self.f = nn.Sequential(
                nn.Linear(dim[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            self.f = nn.Sequential(
                nn.Linear(dim[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, data):
        # x = batch['data']
        return dict(logits=self.f(data))
    
    #This function was not there in libmoon, I added because I needed for PSL Solvers 
    def forward_with_weights(self, x, state_dict):
        """
        x: Tensor of shape [B, dim_in]
        state_dict: dict mapping param names ('f.0.weight', 'f.0.bias', ...) to Tensors
        """
        # On connaît la structure de FullyConnected.f :
        #   [0]=Linear(dim_in→H1), [1]=ReLU, [2]=Linear(H1→H2), [3]=ReLU, [4]=Linear(H2→1)
        # ou avec plus de couches selon l'architecture.
        # Exemple pour l'arch M1 (3 Linear + 2 ReLU) :

        device = x.device

        # Couche 0 : Linear(dim_in → H1)
        w0 = state_dict['f.0.weight'].to(device)
        b0 = state_dict['f.0.bias'].to(device)
        x1 = F.linear(x, weight=w0, bias=b0)
        x1 = F.relu(x1)  # plus de inplace

        # Couche 1 : Linear(H1 → H2)
        w1 = state_dict['f.2.weight'].to(device)
        b1 = state_dict['f.2.bias'].to(device)
        x2 = F.linear(x1, weight=w1, bias=b1)
        x2 = F.relu(x2)

        # Couche 2 (sortie) : Linear(H2 → 1)
        w2 = state_dict['f.4.weight'].to(device)
        b2 = state_dict['f.4.bias'].to(device)
        out = F.linear(x2, weight=w2, bias=b2)

        return out.squeeze(-1)
    
