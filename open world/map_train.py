from data import load_data, load_data_complex
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from fcn_model import mapper

def train_mapper(model_str='transe'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # criterion = nn.MSELoss()
    criterion = nn.PairwiseDistance(p=2)
    batch_size = 128

    if model_str == 'transe':
        x_train, y_train, x_valid, y_valid = load_data(device)
        N_train = x_train.shape[0]
        model = mapper().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

        print("Training mapper function...")
        pbar = tqdm(range(500))
        for epoch in pbar:
            model.train()
            for b in range(int(np.ceil(N_train/batch_size)) ):  
                batch_x = x_train[b*batch_size : (b+1)*batch_size]
                batch_y = y_train[b*batch_size : (b+1)*batch_size]

                optim.zero_grad()
                output = model(torch.from_numpy(batch_x).float().to(device))
                loss = criterion(output, torch.from_numpy(batch_y).float().to(device)).mean()
                loss.backward()
                optim.step()

            if epoch % 5 == 0:
                with torch.no_grad():
                    model.eval()
                    output = model(torch.from_numpy(x_valid).float().to(device))
                    loss = criterion(output, torch.from_numpy(y_valid).float().to(device)).mean()
                    pbar.set_description('loss this epoch %d: %f' % (epoch, loss))

        torch.save(model.state_dict(), './checkpoint/mapper.pt')

    elif model_str == 'complex':
        x_train, yr_train, yi_train, x_valid, yr_valid, yi_valid = load_data_complex(device)
        N_train = x_train.shape[0]
        model_r = mapper().to(device)
        model_i = mapper().to(device)
        optim_r = torch.optim.Adam(model_r.parameters(), lr=0.001)
        optim_i = torch.optim.Adam(model_i.parameters(), lr=0.001)

        print("Training mapper function...")
        pbar = tqdm(range(50))
        for epoch in pbar:
            model_r.train()
            model_i.train()
            for b in range(int(np.ceil(N_train/batch_size)) ):  
                batch_x = x_train[b*batch_size : (b+1)*batch_size]
                batch_yr = yr_train[b*batch_size : (b+1)*batch_size]
                batch_yi = yi_train[b*batch_size : (b+1)*batch_size]

                optim_r.zero_grad()
                optim_i.zero_grad()
                output_r = model_r(torch.from_numpy(batch_x).float().to(device))
                output_i = model_i(torch.from_numpy(batch_x).float().to(device))
                loss_r = criterion(output_r, torch.from_numpy(batch_yr).float().to(device))
                loss_i = criterion(output_i, torch.from_numpy(batch_yi).float().to(device))
                loss_r.backward()
                loss_i.backward()
                optim_r.step()
                optim_i.step()

            with torch.no_grad():
                model_r.eval()
                model_i.eval()
                output_r = model_r(torch.from_numpy(x_valid).float().to(device))
                output_i = model_i(torch.from_numpy(x_valid).float().to(device))
                loss_r = criterion(output_r, torch.from_numpy(yr_valid).float().to(device)).mean()
                loss_i = criterion(output_i, torch.from_numpy(yi_valid).float().to(device)).mean()
                pbar.set_description('loss this epoch %d: %f' % (epoch, loss_r+loss_i))

        torch.save(model_r.state_dict(), './checkpoint/mapper_complex_r.pt')
        torch.save(model_i.state_dict(), './checkpoint/mapper_complex_i.pt')

# train_mapper('transe')


