import torch
from torch import nn

class SimplePSLModel(nn.Module):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.hidden_size = 256
        self.n_obj, self.n_var = problem.n_obj, problem.n_var
        if 'lbound' in dir(problem):
            # The input is a preference vector.
            self.psl_model = nn.Sequential(
                nn.Linear(self.n_obj, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.n_var),
                nn.Sigmoid()
            )

        else:
            self.psl_model = nn.Sequential(
                nn.Linear(self.n_obj, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.n_var),
            )

    def forward(self, pref):
        mid = self.psl_model(pref)
        if 'lbound' in dir(self.problem):

            return mid * torch.Tensor(self.problem.ubound - self.problem.lbound).to(mid.device)  + torch.Tensor(self.problem.lbound).to(mid.device)
        else:
            return mid

#This is the model used for PSL solvers for MTL problems for tabular datas
class SimplePSLModel2(nn.Module):
    def __init__(self, n_obj, n_var, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obj, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, n_var),
        )
    def forward(self, prefs):
        # prefs: [batch, n_obj]
        return self.net(prefs)  # → [batch, n_var]


class PFLModel(nn.Module):
    def __init__(self, n_obj):
        super(PFLModel, self).__init__()
        self.fc1 = nn.Linear(n_obj-1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_obj)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
