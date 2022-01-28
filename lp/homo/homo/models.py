# from models.layers import *
from itertools import combinations
from collections import defaultdict as ddict
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, TAGConv, GATConv
# from models.mlp import MLP
from mlp import MLP
from searchspace import *
from utils import *
from aggregate import *


class GNNModel(nn.Module):
    def get_device(self, args):
        gpu = args.gpu
        return torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    def sample_gumbel(self, shape, args, eps=1e-20):
        U = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, log_alpha, temperature, args):
        y = log_alpha + self.sample_gumbel(log_alpha.size(), args)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, log_alpha, temperature, args, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(log_alpha, temperature, args)

        if not hard:
            return y
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard=y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y, y_hard

    def get_Z_hard(self, log_alpha, temperature, args):
        Z_hard = self.gumbel_softmax(log_alpha, temperature, args, hard=True)[1]
        return Z_hard

    #################################################################################
    def load_searchspace(self):
        self.SearchSpace = SearchSpace()
        self.search_space = self.SearchSpace.search_space
        self.num_choices = self.SearchSpace.num_choices
        self.dims = self.SearchSpace.dims
        self.temperature = 1

        # # self.gcnconv = ['TAGConv', 'GINConv', 'GCNConv', 'SAGEConv', 'GATConv']
        # self.gcnconv = ['GINConv', 'GCNConv', 'SAGEConv', 'GATConv']
        # self.layer_connect = ['stack', 'skip_sum', 'skip_cat']
        # self.layer_agg = ['none', 'cat', 'max_pooling']
        # # self.pool = ['sum', 'diff', 'hadamard', 'max', 'concat']
        # self.pooling = ['sum']

    #################################################################################
    def init_alpha(self):
        # self.log_alpha_gcnconv = nn.Parameter(torch.zeros((self.layers, self.num_choices['gcnconv']),
        #                                                   device=self.device).normal_(mean=1, std=0.01).requires_grad_())
        self.log_alpha_agg = nn.Parameter(torch.zeros((self.layers, self.num_choices['agg']),
                                                          device=self.device).normal_(mean=1, std=0.01).requires_grad_())
        self.log_alpha_combine = nn.Parameter(torch.zeros((self.layers, self.num_choices['combine']),
                                                          device=self.device).normal_(mean=1, std=0.01).requires_grad_())
        self.log_alpha_act = nn.Parameter(torch.zeros((self.layers, self.num_choices['act']),
                                                          device=self.device).normal_(mean=1, std=0.01).requires_grad_())
        self.log_alpha_layer_connect = nn.Parameter(torch.zeros(self.layers, self.num_choices['layer_connect'],
                                                                device=self.device).normal_(mean=1, std=0.01).requires_grad_())
        self.log_alpha_layer_agg = nn.Parameter(torch.zeros(1, self.num_choices['layer_agg'],
                                                            device=self.device).normal_(mean=1, std=0.01).requires_grad_())
        self.log_alpha_pool = nn.Parameter(torch.zeros(1, self.num_choices['pool'],
                                                       device=self.device).normal_(mean=1, std=0.01).requires_grad_())
    #################################################################################
    def update_z_hard(self):
        # self.Z_gcnconv_hard = self.get_Z_hard(self.log_alpha_gcnconv, self.temperature, self.args)
        self.Z_agg_hard = self.get_Z_hard(self.log_alpha_agg, self.temperature, self.args)
        self.Z_combine_hard = self.get_Z_hard(self.log_alpha_combine, self.temperature, self.args)
        self.Z_act_hard = self.get_Z_hard(self.log_alpha_act, self.temperature, self.args)
        self.Z_layer_connect_hard = self.get_Z_hard(self.log_alpha_layer_connect, self.temperature, self.args)
        self.Z_layer_agg_hard = self.get_Z_hard(self.log_alpha_layer_agg, self.temperature, self.args)
        self.Z_pool_hard = self.get_Z_hard(self.log_alpha_pool, self.temperature, self.args)

        # self.Z_hard_dict['gcnconv'].append(self.Z_gcnconv_hard.cpu().tolist())
        self.Z_hard_dict['agg'].append(self.Z_agg_hard.cpu().tolist())
        self.Z_hard_dict['combine'].append(self.Z_combine_hard.cpu().tolist())
        self.Z_hard_dict['act'].append(self.Z_act_hard.cpu().tolist())
        self.Z_hard_dict['layer_connect'].append(self.Z_layer_connect_hard.cpu().tolist())
        self.Z_hard_dict['layer_agg'].append(self.Z_layer_agg_hard.cpu().tolist())
        self.Z_hard_dict['pool'].append(self.Z_pool_hard.cpu().tolist())
        self.Z_hard_dict = dict(self.Z_hard_dict)

    def derive_arch(self):
        for key in self.search_space.keys():
            self.searched_arch_z[key] = self.Z_hard_dict[key][self.max_step]
            self.searched_arch_op[key] = self.z2op(key, self.searched_arch_z[key])
        self.searched_arch_z = dict(self.searched_arch_z)
        self.searched_arch_op = dict(self.searched_arch_op)

        self.Z_agg_hard = torch.tensor(self.searched_arch_z['agg'], device=self.device)
        self.Z_combine_hard = torch.tensor(self.searched_arch_z['combine'], device=self.device)
        self.Z_act_hard = torch.tensor(self.searched_arch_z['act'], device=self.device)
        self.Z_layer_connect_hard = torch.tensor(self.searched_arch_z['layer_connect'], device=self.device)
        self.Z_layer_agg_hard = torch.tensor(self.searched_arch_z['layer_agg'], device=self.device)
        self.Z_pool_hard = torch.tensor(self.searched_arch_z['pool'], device=self.device)

        # self.Z_agg_hard = torch.tensor(self.Z_hard_dict['agg'][self.max_step], device=self.device)
        # self.Z_combine_hard = torch.tensor(self.Z_hard_dict['combine'][self.max_step], device=self.device)
        # self.Z_act_hard = torch.tensor(self.Z_hard_dict['act'][self.max_step], device=self.device)
        # self.Z_layer_connect_hard = torch.tensor(self.Z_hard_dict['layer_connect'][self.max_step], device=self.device)
        # self.Z_layer_agg_hard = torch.tensor(self.Z_hard_dict['layer_agg'][self.max_step], device=self.device)
        # self.Z_pool_hard = torch.tensor(self.Z_hard_dict['pool'][self.max_step], device=self.device)

    def z2op(self, key, z_hard):
        ops = []
        for i in range(len(z_hard)):
            index = z_hard[i].index(1)
            op = self.search_space[key][index]
            ops.append(op)
        return ops


    def load_agg(self):
        self.sum_agg = Sum_AGG(in_channels=self.hidden_features, out_channels=self.hidden_features)
        self.mean_agg = Mean_AGG(in_channels=self.hidden_features, out_channels=self.hidden_features)
        self.max_agg = Max_AGG(in_channels=self.hidden_features, out_channels=self.hidden_features)
    ###################################################################################################################
    ###################################################################################################################

    def __init__(self, layers, in_features, hidden_features, out_features, prop_depth, args, dropout=0.0):
        super(GNNModel, self).__init__()
        self.layers, self.in_features, self.hidden_features, self.out_features, self.prop_depth, self.args, = layers, in_features, hidden_features, out_features, prop_depth, args
        self.device = self.get_device(self.args)
        self.relu = nn.ReLU()
        self.prelu = nn.ModuleList([nn.PReLU() for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

        self.preprocess = nn.Linear(in_features, hidden_features)
        self.linears = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for i in range(layers)])
        self.linears_self = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for i in range(layers)])
        # self.gcn_layers = nn.ModuleList()
        self.combine_merger = nn.ModuleList([nn.Linear(2 * hidden_features, hidden_features) for i in range(layers)])
        self.layer_connect_merger = nn.ModuleList([nn.Linear(2 * hidden_features, hidden_features) for i in range(layers)])
        self.layer_agg_merger = nn.Linear((layers + 1) * hidden_features, hidden_features)
        self.pool_merger = nn.Linear(2 * hidden_features, hidden_features)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for i in range(layers)])
        self.feed_forward = FeedForwardNetwork(hidden_features, out_features)

        # self.init_superlayer()
        self.Z_hard_dict = ddict(list)
        self.searched_arch_z = ddict(list)
        self.searched_arch_op = ddict(list)
        self.load_searchspace()
        self.load_agg()
        self.init_alpha()
        self.update_z_hard()

        self.max_step = None
        self.best_metric_search = None
        self.best_metric_retrain = None


    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        x = self.preprocess(x)
        self.emb_list = [x]

        for i in range(self.layers):
            #######################################################################################################
            # x = self.gcnconv_trans(x, edge_index, layer, self.Z_gcnconv_hard[i].view(1, -1))
            x_self, x_n = self.linears_self[i](x), self.linears[i](x)
            x_n = self.agg_trans(x_n, edge_index, self.Z_agg_hard[i].view(1, -1))
            x = self.combine_trans(i, x_self, x_n, self.Z_combine_hard[i].view(1, -1))
            #######################################################################################################
            # x = self.act(x)
            x = self.act_trans(i, x, self.Z_act_hard[i].view(1, -1))
            x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
            x = self.layer_norms[i](x)

            self.emb_list.append(x)
            x = self.layer_connect_trans(i+1, self.Z_layer_connect_hard[i].view(1, -1))
        x = self.layer_agg_trans(self.Z_layer_agg_hard)
        self.emb_list = []

        x = self.get_minibatch_embeddings(x, batch)
        x = self.feed_forward(x)
        return x

    def get_minibatch_embeddings(self, x, batch):
        device = x.device
        set_indices, batch, num_graphs = batch.set_indices, batch.batch, batch.num_graphs
        num_nodes = torch.eye(num_graphs)[batch].to(device).sum(dim=0)
        zero = torch.tensor([0], dtype=torch.long).to(device)
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1]])
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
        assert (index_bases.size(0) == set_indices.size(0))
        set_indices_batch = index_bases + set_indices
        # # print('set_indices shape', set_indices.shape, 'index_bases shape', index_bases.shape, 'x shape:', x.shape)
        # print(set_indices_batch.shape)
        # print(x.shape)
        x = x[set_indices_batch]  # shape [B, set_size, F], set_size=1, 2, or 3 for node, link and tri
        print(x.shape)
        # x = self.pool(x)
        x = self.pool_trans(x, self.Z_pool_hard)
        return x

    def agg_trans(self, x_n, edge_index, z_hard):
        y = []
        for agg in [self.sum_agg, self.mean_agg, self.max_agg]:
            temp = agg(x_n, edge_index)
            y.append(temp)
        x_n = torch.stack(y, dim=0)
        x_n = torch.einsum('ij,jkl -> ikl', z_hard, x_n).squeeze(0)
        return x_n

    def combine_trans(self, cur_layer, x, x_n, z_hard):
        y = []
        for combine in self.search_space['combine']:
            temp = self.combine_map(cur_layer, x, x_n, combine)
            y.append(temp)
        x = torch.stack(y, dim=0)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def combine_map(self, cur_layer, x, x_n, combine):
        if combine == 'sum':
            x = x + x_n
        if combine == 'concat':
            x = torch.cat([x, x_n], axis=-1)
            x = self.combine_merger[cur_layer](x)
        return x

    def act_trans(self, cur_layer, x, z_hard):
        y = []
        for act in self.search_space['act']:
            temp = self.act_map(cur_layer, x, act)
            y.append(temp)
        x = torch.stack(y, dim=0)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def act_map(self, cur_layer, x, act):
        if act == 'relu':
            x = self.relu(x)
        if act == 'prelu':
            x = self.prelu[cur_layer](x)
        return x

    def layer_connect_trans(self, cur_layer, z_hard):
        y = []
        for layer_connect in self.search_space['layer_connect']:
            temp = self.layer_connect_map(cur_layer, layer_connect)
            y.append(temp)
        x = torch.stack(y, dim=0)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def layer_connect_map(self, cur_layer, layer_connect):
        if layer_connect == 'stack':
            x = self.emb_list[-1]
        if layer_connect == 'skip_sum':
            x = self.emb_list[-1] + self.emb_list[-2]
        if layer_connect == 'skip_cat':
            x = torch.cat([self.emb_list[-2], self.emb_list[-1]], dim=-1)
            x = self.layer_connect_merger[cur_layer-1](x)
        return x


    def layer_agg_trans(self, z_hard):
        y = []
        for layer_agg in self.search_space['layer_agg']:
            #             temp = self.layer_agg_map(x, layer_agg)
            temp = self.layer_agg_map(layer_agg)
            y.append(temp)
        x = torch.stack(y, dim=0)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def layer_agg_map(self, layer_agg):
        if layer_agg == 'none':
            x = self.emb_list[-1]
        if layer_agg == 'max_pooling':
            # x = torch.stack(self.emb_list).to(self.device)
            x = torch.stack(self.emb_list)
            x = x.max(dim=0)[0]
        if layer_agg == 'concat':
            # x = torch.cat(self.emb_list, dim=-1).to(self.device)
            x = torch.cat(self.emb_list, dim=-1)
            x = self.layer_agg_merger(x)
        return x

    def pool_trans(self, x, z_hard):
        y = []
        for pool in self.search_space['pool']:
            temp = self.pool_map(x, pool)
            y.append(temp)
        x = torch.stack(y, dim=0)
        print(z_hard.shape)
        print(x.shape)
        x = torch.einsum('ij,jkl -> ikl', z_hard, x).squeeze(0)
        return x

    def pool_map(self, x, pool):
        if pool == 'sum':
            x = x.sum(dim=1)
        if pool == 'max':
            x = x.max(dim=1)[0]
        # if pool == 'diff':
        #     x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
        #     for i, j in combinations(range(x.size(1)), 2):
        #         x_diff += torch.abs(x[:, i, :] - x[:, j, :])
        #         x = x_diff
        if pool == 'concat':
            x = x.view(x.shape[0], -1)
            x = self.pool_merger(x)
        return x

    # def pool(self, x):
    #     if x.size(1) == 1:
    #         return torch.squeeze(x, dim=1)
    #     # use mean/diff/max to pool each set's representations
    #     x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
    #     for i, j in combinations(range(x.size(1)), 2):
    #         x_diff += torch.abs(x[:, i, :] - x[:, j, :])
    #     x_mean = x.mean(dim=1)
    #     x_max = x.max(dim=1)[0]
    #     x = self.merger(torch.cat([x_diff, x_mean, x_max], dim=-1))
    #     return x


    def short_summary(self):
        return 'Model: Auto-GNN, #layers: {}, in_features: {}, hidden_features: {}, out_features: {}'.format(self.layers,
                                                                                                       self.in_features,
                                                                                                       self.hidden_features,
                                                                                                       self.out_features)


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), dropout=0):
        super(FeedForwardNetwork, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout)
        # self.layer2 = nn.Sequential(nn.Linear(in_features, out_features), nn.LogSoftmax(dim=-1))
        self.layer2 = nn.Linear(in_features, out_features)


    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x
