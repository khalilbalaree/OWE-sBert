import torch.nn as nn

class mapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768,300, bias=True)
    
    def forward(self, encoded):
        return self.linear1(encoded)