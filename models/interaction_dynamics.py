from torch import nn
import torch
import torch.nn.functional as F

from utils.build import make_optimizer


class IntNetwork(nn.Module):
    def __init__(self, hidden_dim, inp_dim):
        super().__init__()
        self.encode_dim = nn.Linear(6 * inp_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.decode_1 = nn.Linear(hidden_dim, hidden_dim)
        self.decode_2 = nn.Linear(hidden_dim, inp_dim)


    def forward(self,  data):
        # Data should be 3 dimensional, of B x O x E
        data = torch.cat(data, dim=-1)
        obj = data.size(1)
        data_e1 = data[:, :, None, :].repeat(1, 1, obj, 1)
        data_e2 = data[:, None, :, :].repeat(1, obj, 1, 1)

        data_cat = torch.cat([data_e1, data_e2], dim=3)

        output_emb = self.fc3(F.relu(self.fc2(F.relu(self.fc1(F.relu(self.encode_dim(data_cat)))))))
        output_emb = output_emb.mean(dim=2)

        decode = self.decode_2(F.relu(self.decode_1(output_emb)))

        return decode


class InteractionDynamics(nn.Module):
    def __init__(self, cfg, dynamics_dataset):
        super().__init__()
        self.input_size = dynamics_dataset.input_size
        self.int_net = IntNetwork(cfg.MODEL.HIDDEN_SIZE*2, self.input_size // dynamics_dataset.max_num_objects)

        linear = nn.Linear(cfg.MODEL.HIDDEN_SIZE*2, self.input_size)

        self.linear = nn.Sequential(nn.ReLU(),linear)
        self.dataset = dynamics_dataset
        self.past_frame = 3


    def forward(self,  data, match=False):
        input = data["input"]
        timesteps = input.size(1)
        bs = input.size()

        outputs = []
        past_states = [torch.zeros_like(input[:, 0, :, :]) for i in range(self.past_frame)]
        state = past_states[-1]
        magic_penalty = [0 for i in range(bs[0])]

        for t in range(timesteps):
            if match:
                exist = input[:, t, :, 1:2]

                if  t == 0:
                    state = exist * input[:, t] + (1 - exist) * state
                    state_exist = exist
                else:
                    state = past_states[-1]
                    trans = input[:, t, :, 23:26]
                    state_trans = state[:, :, 23:26]
                    bs = input.size()

                    min_tresh = 6.
                    for i in range(bs[0]):
                        select_trans = trans[i, exist[i].bool().squeeze()]
                        select_state_trans = state_trans[i, state_exist[i].bool().squeeze()]

                        dist = select_trans[None, :, :] - select_state_trans[:, None, :]
                        dist = torch.norm(dist, p=2, dim=2)

                        min_dist, min_dist_idx = dist.min(dim=1)
                        min_dist = min_dist.detach().cpu().numpy()
                        min_dist_idx = min_dist_idx.detach().cpu().numpy()

                        for j in range(min_dist.shape[0]):
                            if min_dist[j] > min_tresh:
                                # store state as new entry
                                magic_penalty[i] = float(magic_penalty[i] + 30)
                                nelem = state_exist[i].sum().long()
                                nelem = nelem % state_exist.size(1)
                                state_exist[i, nelem, :] = 1
                                state[i,  nelem, :] = input[i, t, j]
                            else:
                                state[i, min_dist_idx[j], :] = input[i, t, j]
                                magic_penalty[i] = float(magic_penalty[i] + min_dist[j].item())

                past_states = past_states[1:] + [state]
                state = self.int_net(past_states)
                outputs.append(state)

                if t % 5 == 0:
                    state = state.detach()

            else:
                exist = input[:, t, :, 1:2]
                state = past_states[-1]
                state = exist * input[:, t] + (1 - exist) * state

                past_states = past_states[1:] + [state]
                state = self.int_net(past_states)
                outputs.append(state)

                if t % 5 == 0:
                    state = state.detach()

        outputs = torch.stack(outputs[:-1], dim=1)
        output = self.dataset.input_2_dict(outputs)

        # if self.training:
        #     loss_dict = self.dataset.loss(output,data["targets"])
        # else:
        #     loss_dict = 0
        target_dict = {}
        for k, v in data["targets"].items():
            target_dict[k] = data["targets"][k][:, 1:].contiguous()

        loss_dict = self.dataset.loss(output, target_dict)

        return {"output":output,
                "loss_dict":loss_dict, "magic_penalty": magic_penalty}
