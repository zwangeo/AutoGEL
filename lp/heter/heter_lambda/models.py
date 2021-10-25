from helper import *
from compgcn_conv import CompGCNConv
from compgcn_conv_basis import CompGCNConvBasis
from collections import defaultdict as ddict
from torch import nn
from searchspace import *


class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)


class CompGCNBase(BaseModel):
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

	def layer_agg_trans(self, z_hard, emb='x'):
		y = []
		for layer_agg in self.search_space['layer_agg']:
			temp = self.layer_agg_map(layer_agg, emb)
			y.append(temp)
		out = torch.stack(y, dim=0)
		out = torch.einsum('ij,jkl -> ikl', z_hard, out).squeeze(0)
		return out

	def layer_agg_map(self, layer_agg, emb):
		if emb == 'x':
			if layer_agg == 'stack':
				x = self.emb_list_x[-1]
			if layer_agg == 'sum':
				# x = self.emb_list_x[-1] + self.emb_list_x[-2]
				x = torch.stack(self.emb_list_x)
				x = x.sum(dim=0)
			if layer_agg == 'concat':
				x = torch.cat(self.emb_list_x, dim=-1)
				x = self.layer_agg_merger_x(x)
			if layer_agg == 'max_pooling':
				x = torch.stack(self.emb_list_x)
				x = x.max(dim=0)[0]
			return x

		if emb == 'r':
			if layer_agg == 'stack':
				r = self.emb_list_r[-1]
			if layer_agg == 'sum':
				r = torch.stack(self.emb_list_r)
				r = r.sum(dim=0)
			if layer_agg == 'concat':
				r = torch.cat(self.emb_list_r, dim=-1)
				r = self.layer_agg_merger_r(r)
			if layer_agg == 'max_pooling':
				r = torch.stack(self.emb_list_r)
				r = r.max(dim=0)[0]
			return r

	#################################################################################
	def load_searchspace(self):
		self.SearchSpace = SearchSpace()
		self.search_space = self.SearchSpace.search_space
		self.num_choices = self.SearchSpace.num_choices
		self.dims = self.SearchSpace.dims
		self.temperature = 1
	#
	def init_alpha(self):
		self.log_alpha_layer_agg = nn.Parameter(torch.zeros((1, self.num_choices['layer_agg']),
															device=self.device).normal_(mean=1,
																						std=0.01).requires_grad_())

	def update_z_hard(self, epoch):
		################################################################################################################
		# micro update
		if self.p.gcn_layer == 1:
			self.conv1.update_z_hard_micro(epoch)
			self.Z_hard_dict['agg'].append(self.conv1.Z_agg_hard.cpu().tolist())
			self.Z_hard_dict['combine'].append(self.conv1.Z_combine_hard.cpu().tolist())
			self.Z_hard_dict['comp'].append(self.conv1.Z_comp_hard.cpu().tolist())

		if self.p.gcn_layer == 2:
			self.conv1.update_z_hard_micro(epoch)
			self.conv2.update_z_hard_micro(epoch)
			self.Z_hard_dict['agg'].append(self.conv1.Z_agg_hard.cpu().tolist() + self.conv2.Z_agg_hard.cpu().tolist())
			self.Z_hard_dict['combine'].append(self.conv1.Z_combine_hard.cpu().tolist() + self.conv2.Z_combine_hard.cpu().tolist())
			self.Z_hard_dict['comp'].append(self.conv1.Z_comp_hard.cpu().tolist() + self.conv2.Z_comp_hard.cpu().tolist())

		################################################################################################################
		# macro update
		if self.p.macro_search:
			self.Z_layer_agg_hard = self.get_Z_hard(self.log_alpha_layer_agg, self.temperature, self.p)
			if epoch > 30 and epoch % 5 == 0 and self.temperature >= 1e-20:
				self.temperature *= 1e-1
			self.Z_hard_dict['layer_agg'].append(self.Z_layer_agg_hard.cpu().tolist())

		self.Z_hard_dict = dict(self.Z_hard_dict)


	def derive_arch(self):
		# for key in self.search_space.keys():
		for key in self.Z_hard_dict.keys():
			self.searched_arch_z[key] = self.Z_hard_dict[key][self.best_epoch]
			self.searched_arch_op[key] = self.z2op(key, self.searched_arch_z[key])
		self.searched_arch_z = dict(self.searched_arch_z)
		self.searched_arch_op = dict(self.searched_arch_op)

		self.conv1.Z_agg_hard = torch.tensor(self.conv1.Z_hard_dict_micro['agg'][self.best_epoch], device=self.device)
		self.conv1.Z_combine_hard = torch.tensor(self.conv1.Z_hard_dict_micro['combine'][self.best_epoch], device=self.device)
		self.conv1.Z_comp_hard = torch.tensor(self.conv1.Z_hard_dict_micro['comp'][self.best_epoch], device=self.device)
		if self.p.gcn_layer == 2:
			self.conv2.Z_agg_hard = torch.tensor(self.conv2.Z_hard_dict_micro['agg'][self.best_epoch], device=self.device)
			self.conv2.Z_combine_hard = torch.tensor(self.conv2.Z_hard_dict_micro['combine'][self.best_epoch], device=self.device)
			self.conv2.Z_comp_hard = torch.tensor(self.conv2.Z_hard_dict_micro['comp'][self.best_epoch], device=self.device)

		if self.p.macro_search:
			self.Z_layer_agg_hard = torch.tensor(self.Z_hard_dict['layer_agg'][self.best_epoch], device=self.device)


	def z2op(self, key, z_hard):
		ops = []
		for i in range(len(z_hard)):
			index = z_hard[i].index(1)
			op = self.search_space[key][index]
			ops.append(op)
		return ops


	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		self.device		= self.edge_index.device

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

		# # self.load_searchspace()
		#
		# # self.preprocess_x = nn.Linear(self.p.init_dim, self.p.gcn_dim)
		# # self.preprocess_r = nn.Linear(self.p.init_dim, self.p.gcn_dim)
		# # self.layer_connect_merger_x = nn.ModuleList(nn.Linear(2*self.p.gcn_dim, self.p.gcn_dim) for i in range(self.p.gcn_layer))
		# # self.layer_connect_merger_r = nn.ModuleList(nn.Linear(2*self.p.gcn_dim, self.p.gcn_dim) for i in range(self.p.gcn_layer))
		self.layer_agg_merger_x = nn.Linear((self.p.gcn_layer+1)*self.p.gcn_dim, self.p.gcn_dim)
		self.layer_agg_merger_r = nn.Linear((self.p.gcn_layer+1)*self.p.gcn_dim, self.p.gcn_dim)

		self.Z_hard_dict = ddict(list)
		self.searched_arch_z = ddict(list)
		self.searched_arch_op = ddict(list)
		self.load_searchspace()
		# self.init_alpha()
		self.update_z_hard(epoch=0)

		self.best_epoch = None


	def forward_base(self, sub, rel, drop1, drop2):
		x = self.init_embed
		r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		self.emb_list_x = [x]
		self.emb_list_r = []

		x, r = self.conv1(self.emb_list_r, x, self.edge_index, self.edge_type, rel_embed=r, first_layer=True)
		x = drop1(x)
		self.emb_list_x.append(x)
		self.emb_list_r.append(r)
		if self.p.gcn_layer == 2:
			x, r = self.conv2(self.emb_list_r, x, self.edge_index, self.edge_type, rel_embed=r, first_layer=False)
			x = drop2(x)
			self.emb_list_x.append(x)
			self.emb_list_r.append(r)

		if self.p.macro_search:
			x = self.layer_agg_trans(self.Z_layer_agg_hard, emb='x')
			r = self.layer_agg_trans(self.Z_layer_agg_hard, emb='r')
		self.emb_list_x = []
		self.emb_list_r = []

		sub_emb = torch.index_select(x, 0, sub)
		rel_emb = torch.index_select(r, 0, rel)
		return sub_emb, rel_emb, x


class Auto_CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

		# self.best_epoch = None
		# # self.z_hard_dict = ddict(list)


	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score
