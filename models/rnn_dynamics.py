from torch import nn

from utils.build import make_optimizer


class RNNDynamics(nn.Module):
    def __init__(self, cfg, dynamics_dataset):
        super().__init__()
        self.input_size = dynamics_dataset.input_size
        self.gru = nn.GRU(input_size=self.input_size,
                       hidden_size= cfg.MODEL.HIDDEN_SIZE,
                       num_layers=cfg.MODEL.RNN_NUM_LAYERS,
                       batch_first= True,
                       dropout=cfg.MODEL.DROP_OUT,
                       bidirectional=True)

        linear = nn.Linear(cfg.MODEL.HIDDEN_SIZE*2, self.input_size)

        self.linear = nn.Sequential(nn.ReLU(),linear)
        self.dataset = dynamics_dataset


    def forward(self,  data):
        input = data["input"]
        x = input.view(input.shape[0],
                           input.shape[1],
                           -1)
        x, _ = self.gru(x)
        x = self.linear(x)
        x = x.view(*data["input"].shape)
        output = self.dataset.input_2_dict(x)
        # if self.training:
        #     loss_dict = self.dataset.loss(output,data["targets"])
        # else:
        #     loss_dict = 0
        loss_dict = self.dataset.loss(output, data["targets"])
        return {"output":output,
                "loss_dict":loss_dict}
