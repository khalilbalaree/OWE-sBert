import torch
import os
from torch._C import device
import torch.nn as nn
import numpy as np
from tqdm.std import tqdm
from data import load_model, load_open_word_test

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

model_path = './checkpoint/mapper.pt'
if not os.path.exists(model_path):
    exit("Train mapper first!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mapper().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

hs, ts, rs = load_open_word_test(device)
count = 0
print('Testing...')
for i in tqdm(range(len(hs))):
    output = model(torch.from_numpy(hs[i]).float().to(device))
    output = output.detach().cpu().numpy()

    e,r = load_model()

    i_j = {}
    minimum = 1e3
    for j, ee in enumerate(e):
        d = np.sum(np.abs(output+r[int(rs[i])]-ee))
        i_j[str(j)] = d

    sorted_d = {k: v for k, v in sorted(i_j.items(), key=lambda item: item[1])}

    hit = 10
    if ts[i] in list(sorted_d)[:hit]:
        count += 1
        
print(count/len(hs))
    


# nlp = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)
# text = ''
# this_x = nlp.encode(text)
# output = model(torch.from_numpy(this_x).float().to(device))
# output = output.detach().cpu().numpy()

# e,r = load_model()



# i_j = {}
# for j, ee in enumerate(e):
#     d = np.sum(np.abs(output+r[71]-ee))
#     # d = np.linalg.norm()
#     i_j[j] = d

# sorted_d = {k: v for k, v in sorted(i_j.items(), key=lambda item: item[1])}

# entities = load_entities()
# for i in list(sorted_d)[:10]:
#     print(entities[str(i)])










