import bert_model
import numpy as np
from tqdm import tqdm
import torch
from data import load_data_for_bert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train, y_train, x_valid, y_valid = load_data_for_bert()
N_train = len(x_train)

model = bert_model.mapper().to(device)
tokenizer = bert_model.text_tokens()

optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = torch.nn.PairwiseDistance(p=2)
batch_size = 10

print("Training mapper function...")
pbar = tqdm(range(10))
for epoch in pbar:
    model.train()
    for b in range(int(np.ceil(N_train/batch_size)) ):  
        batch_x = x_train[b*batch_size : (b+1)*batch_size]
        batch_y = y_train[b*batch_size : (b+1)*batch_size]

        output = model(tokenizer.get_tokens(batch_x).to(device))
        loss = criterion(output, torch.from_numpy(batch_y).float().to(device)).mean()
        loss.backward()
        if (b+1)%10 == 0:
            # every 10 iterations of batches of size 10
            optim.step()
            optim.zero_grad()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            output = model(tokenizer.get_tokens(x_valid).to(device))
            loss = criterion(output, torch.from_numpy(y_valid).float().to(device)).mean()
            pbar.set_description('loss this epoch %d: %f' % (epoch, loss))

torch.save(model.state_dict(), './checkpoint/bert_mapper.pt')