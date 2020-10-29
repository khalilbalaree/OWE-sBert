from data import load_data
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class mapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768,500)
        self.drop_layer = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(500,300)
    
    def forward(self, encoded):
        a = self.linear1(encoded)
        b = self.drop_layer(a)
        return self.linear2(b)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train, y_train, x_valid, y_valid = load_data(device)

model = mapper().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
criterion = nn.MSELoss()

batch_size = 10
N_train = x_train.shape[0]

print("Training mapper function...")
pbar = tqdm(range(20))
for epoch in pbar:
    model.train()
    for b in range(int(np.ceil(N_train/batch_size)) ):  
        batch_x = x_train[b*batch_size : (b+1)*batch_size]
        batch_y = y_train[b*batch_size : (b+1)*batch_size]

        optim.zero_grad()
        output = model(torch.from_numpy(batch_x).float().to(device))
        loss = criterion(output, torch.from_numpy(batch_y).float().to(device))
        loss.backward()
        optim.step()

    with torch.no_grad():
        model.eval()
        output = model(torch.from_numpy(x_valid).float().to(device))
        loss = criterion(output, torch.from_numpy(y_valid).float().to(device))
        pbar.set_description('loss this epoch %d: %f' % (epoch, loss))

torch.save(model.state_dict(), './checkpoint/mapper.pt')
