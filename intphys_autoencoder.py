import torch.nn as nn
import torch
import torch.optim as optim
import torch.functional as F
import json_dataloader
from json_dataloader import tensorToAttributes
from utils.misc import setup_cfg
from run_experiment import parse_args
import time

class IntphysAE(nn.Module):
    def __init__(self):
        super(IntphysAE, self).__init__()
        self.encoder = nn.Sequential(
                 nn.Linear(33, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32,16),
                nn.ReLU(inplace=True),
                nn.Linear(16,16),
                nn.ReLU(inplace=True)
                )
        self.decoder = nn.Sequential(
                nn.Linear(16, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32,33),
                )

    def forward(self, x):
        encoding = self.encoder.forward(x)
        return self.decoder.forward(encoding)

def train(cfg, epochs=2):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = IntphysAE().to(device)
    # model.to(device)

    #train_data = json_dataloader.IntphysJsonTensor(cfg, "_train")
    #val_data = json_dataloader.IntphysJsonTensor(cfg, "_val")

    #Test block using split of validation set b/c full training seet takes too long
    json_data = json_dataloader.IntphysJsonTensor(cfg, "_val")
    train_size = round(len(json_data)*.8)
    val_size = len(json_data) - train_size
    train_data, val_data = torch.utils.data.random_split(json_data, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=32, shuffle=True,
            num_workers=8)

    #dataset_utils = json_dataloader.DatasetUtils(val_data)
    dataset_utils = json_dataloader.DatasetUtils(json_data)
    optimizer = optim.Adam(model.parameters(), lr=.0001)
    criterion = dataset_utils.loss
    train_loss = []
    val_loss = []

    for epoch in range(1, epochs+1):
        running_loss = 0
        start = time.time()
        for i, data in enumerate(train_loader):
            data = dataset_utils.normalize(data.to(device))
            recons  = model.forward(data)
            loss = criterion(recons, data)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and i!=0:
                running_loss/=100
                train_loss.append(running_loss)
                with torch.no_grad():
                    v_loss = 0
                    for j, data in enumerate(val_loader):
                        data = dataset_utils.normalize(data.to(device))
                        recons  = model.forward(data)
                        loss = criterion(recons, data)
                        v_loss += loss.item()
                    v_loss /= len(val_loader)
                    val_loss.append(v_loss)

                print("\nIteration {0}/{1} of Epoch {2}\nRunning Loss: {3}\nValidatiion Loss: {4}\n{5} seconds".format(i, len(train_loader), epoch, running_loss, v_loss, time.time()-start))
                running_loss=0
                start = time.time()

    return model


def main(args):
    cfg = setup_cfg(args, args.distributed)
    model = train(cfg)
    return model

"""with torch.no_grad():
        te_r = val_data[0]
        d_r = tensorToAttributes(te_r)
        re_t = model.forward(te_r)
        re_d = tensorToAttributes(re_t)"""
if __name__ == "__main__":
    args = parse_args()
    model = main(args)





















