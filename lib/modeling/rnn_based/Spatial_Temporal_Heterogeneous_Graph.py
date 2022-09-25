import copy
import torch
from torch import nn
import torch.nn.functional as F
import math


class Spatial_Temporal_Heterogeneous_Graph(nn.Module):

    def __init__(self, cfg, v_len=5, sub_layers=3):

        super(Spatial_Temporal_Heterogeneous_Graph, self).__init__()

        self.cfg = cfg

        self.batch_size = self.cfg.SOLVER.BATCH_SIZE
        if self.cfg.DATASET.NAME == 'PIE':
            self.sub_graph = nn.ModuleDict({
                'x_neighbor': SubGraph(layers_number=sub_layers, v_len=5),
                'x_light': SubGraph(layers_number=sub_layers, v_len=6),
                'x_sign': SubGraph(layers_number=sub_layers, v_len=5),
                'x_crosswalk': SubGraph(layers_number=sub_layers, v_len=4),
                'x_station': SubGraph(layers_number=sub_layers, v_len=4),
                'x_ego': SubGraph(layers_number=sub_layers, v_len=4),
            })

            self.traffic_embedding = nn.ModuleDict({
                'x_neighbor': nn.Sequential(nn.Linear(5, 64), nn.ReLU(),
                                            nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1)),
                'x_light': nn.Sequential(nn.Linear(6, 64), nn.ReLU(),
                                         nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1)),
                'x_sign': nn.Sequential(nn.Linear(5, 64), nn.ReLU(),
                                        nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1)),
                'x_crosswalk': nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                             nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1)),
                'x_station': nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                           nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1)),
                'x_ego': nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                       nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1))
            })
            self.traffic_keys = self.cfg.MODEL.TRAFFIC_KEYS
        else:
            self.sub_graph = nn.ModuleDict({
                'x_neighbor': SubGraph(layers_number=sub_layers, v_len=4),
                'x_light': SubGraph(layers_number=sub_layers, v_len=1),
                'x_sign': SubGraph(layers_number=sub_layers, v_len=2),
                'x_crosswalk': SubGraph(layers_number=sub_layers, v_len=1),
                'x_ego': SubGraph(layers_number=sub_layers, v_len=1)

            })

            self.traffic_embedding = nn.ModuleDict({
                'x_neighbor': nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                            nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1)),
                'x_light': nn.Sequential(nn.Linear(1, 64), nn.ReLU(),
                                         nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1)),
                'x_sign': nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                        nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1)),
                'x_crosswalk': nn.Sequential(nn.Linear(1, 64), nn.ReLU(),
                                             nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1)),
                'x_ego': nn.Sequential(nn.Linear(1, 64), nn.ReLU(),
                                       nn.GRU(input_size=64, hidden_size=64, batch_first=True, dropout=0.1))
            })
            self.traffic_keys = ['x_neighbor', 'x_crosswalk', 'x_light', 'x_sign']

        self.p_len = v_len * (2 ** sub_layers)

    def forward(self, x_traffic, x_bbox, t):

        all_traffic_features = []
        num_traffics = {}
        all_traffic_attentions = {}
        for k in self.traffic_keys:

            if isinstance(x_traffic[k], torch.Tensor):
                x_traffic_k, _ = self.traffic_embedding[k](x_traffic[k][:, :t + 1].to(x_bbox.device))
                all_traffic_features.append(x_traffic_k[:, -1])

            elif isinstance(x_traffic[k], list):
                x_traffic_k = torch.cat(x_traffic[k], dim=0).to(x_bbox.device)  # sum_num,30,5

                if len(x_traffic_k) > 0:

                    traffic_cls = 'cls_' + k.split('_')[-1]

                    batch_size = len(x_traffic[k])
                    num_traffics[k] = [len(v) if len(v) > 0 else 0 for v in
                                       x_traffic[k]]
                    num_objects = sum(num_traffics[k])

                    # masks = (torch.cat(x_traffic[traffic_cls], dim=0) != -1).to('cuda:0')  # sum_num,30,1
                    # masks = masks[:, t] if len(masks) > 0 else masks  # sum_num,1

                    batch_traffic_id_map = torch.zeros(batch_size, num_objects).to(x_bbox.device)
                    indices = torch.repeat_interleave(torch.tensor(range(batch_size)),
                                                      torch.tensor(num_traffics[k])).to(x_bbox.device)
                    batch_traffic_id_map[indices, range(num_objects)] = 1

                    x_traffic_k = x_traffic_k  # sum_num,5
                    x_traffic_k_embed,attention = self.sub_graph[k](x_traffic_k, x_bbox, batch_traffic_id_map, num_traffics[k], t)
                    all_traffic_features.append(x_traffic_k_embed)
                    all_traffic_attentions[k] = attention
                    # print(x_traffic_k_embed.unsqueeze(1).shape)

                else:
                    all_traffic_features.append(torch.zeros((x_bbox.shape[0], 64)).to(x_bbox.device))
            else:
                raise TypeError("traffic type unknown: " + type(x_traffic[k]))

        all_traffic_features_embed = torch.cat(all_traffic_features, dim=-1)


        return all_traffic_features_embed, all_traffic_attentions


class SubGraph(nn.Module):

    def __init__(self, v_len, layers_number):

        super(SubGraph, self).__init__()

        self.layers=nn.Sequential()
        self.L=layers_number

        self.layers.add_module("sub{}".format(0),SubGraphLayer(v_len,8))

        if self.L>1:
            for i in range(1,self.L):
                self.layers.add_module("sub{}".format(i),SubGraphLayer(2**(i+3),2**(i+3)))

        self.x_bbox_embed = MLP(4, 2**(self.L+3))
        self.x_bbox_gru = nn.GRU(input_size=4, hidden_size=64, batch_first=True, dropout=0.1)
        self.gru = nn.GRU(input_size=2**(self.L+3), hidden_size=64, batch_first=True, dropout=0.1)

        self.global_graph = GlobalGraph(2**(self.L+3))

        self.v_len = v_len
        self.layers_number = layers_number

    def forward(self, x, x_bbox, batch_traffic_id_map, num_traffics, t):

        # assert len(x.shape) == 3
        batch_size = x.shape[0]
        for i in range(self.L):
            x = self.layers[i](x, batch_traffic_id_map,torch.tensor(num_traffics).to(x_bbox.device))  # [batch_size, v_number, p_len]

        x_bbox_embed = self.x_bbox_embed(x_bbox)

        x,A = self.global_graph(x_bbox_embed, x, x, batch_traffic_id_map)

        x, _ = self.gru(x[:, :t + 1])

        x = x[:, -1]


        return x,A


class SubGraphLayer(nn.Module):
    r"""
    One layer of subgraph, include the MLP of g_enc.
    The calculation detail in this paper's 3.2 section.
    Input some vectors with 'len' length, the output's length is '2*len'(because of
    concat operator).
    """

    def __init__(self, len_in, len_out):

        super(SubGraphLayer, self).__init__()
        self.g_enc = MLP(len_in, len_out)

    def forward(self, x, batch_traffic_id_map, num_traffics):

        # assert len(x.shape) == 3
        x = self.g_enc(x)

        # avgpool1d
        x2 = torch.matmul(batch_traffic_id_map, x.transpose(0, 1))  # sum_nonzero,len
        num_traffics_tensor = num_traffics.unsqueeze(-1)
        num_traffics_tensor = torch.where(num_traffics_tensor == 0, torch.ones_like(num_traffics_tensor),
                                          num_traffics_tensor)
        x2 = x2 / num_traffics_tensor  # sum_nonzero,len
        x2 = torch.repeat_interleave(x2, num_traffics, dim=1)

        y = torch.cat((x2.transpose(0, 1), x), dim=2)
        # assert y.shape == (batch_size, n, length * 2)
        return y


class MLP(nn.Module):
    r"""
    Construct a MLP, include a single fully-connected layer,
    followed by layer normalization and then ReLU.
    """

    def __init__(self, input_size, output_size, hidden_size=64):
        r"""
        self.norm is layer normalization.
        Args:
            input_size: the size of input layer.
            output_size: the size of output layer.
            hidden_size: the size of output layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r"""
        Args:
            x: x.shape = [batch_size, ..., input_size]
        """
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GlobalGraph(nn.Module):

    def __init__(self, hidden_size=64):
        super(GlobalGraph, self).__init__()

        self.hidden_size = hidden_size
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, batch_traffic_id_map):
        num_objects = k.shape[0]
        q = self.q_linear(q).transpose(0, 1)
        k = self.k_linear(k).transpose(0, 1)
        v = self.v_linear(v).transpose(0, 1)

        a = torch.matmul(q, k.transpose(1, 2))
        a = a / math.sqrt(self.hidden_size)
        a = torch.exp(a)
        a = torch.mul(a, batch_traffic_id_map)
        a_sum = torch.sum(a, dim=-1)
        a_sum = torch.where(a_sum == 0, torch.ones_like(a_sum),
                            a_sum)
        A = torch.div(a, a_sum.unsqueeze(-1).repeat(1, 1, num_objects))

        return self.dropout(torch.matmul(A, v).transpose(0, 1)),A


if __name__ == '__main__':
    a = torch.randn((5, 15, 5)).to('cuda:0')
    b = torch.randn((3, 15, 5)).to('cuda:0')
    c = torch.randn((7, 15, 5)).to('cuda:0')
    d = torch.randn((2, 15, 5)).to('cuda:0')

    x_traffic = {}
    x_traffic['x_neighbor'] = [a, d, c, d]

    cfg = 1
    t = 3
    vectornet = VectorNet(dataset='PIE').to('cuda:0')
    vectornet(x_traffic, t)
