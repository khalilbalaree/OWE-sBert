import torch.nn as nn

class mapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768,512)
        self.linear2 = nn.Linear(512, 300)
    
    def forward(self, encoded):
        return self.linear2(self.linear1(encoded))