from collections import defaultdict as ddict
from helper import *
from message_passing import MessagePassing
from torch import nn


class CompGCNConvBasis(MessagePassing):
	def init_alpha_micro(self):

		self.log_alpha_agg = nn.Parameter(torch.zeros((1, self.num_choices['agg']),
													  device=self.device).normal_(mean=1, std=0.01).requires_grad_())
		self.log_alpha_combine = nn.Parameter(torch.zeros((1, self.num_choices['combine']),
														  device=self.device).normal_(mean=1,
																					  std=0.01).requires_grad_())
		self.log_alpha_comp = nn.Parameter(torch.zeros(1, self.num_choices['comp'],
													   device=self.device).normal_(mean=1, std=0.01).requires_grad_())


	def update_z_hard_micro(self, epoch):
		self.Z_agg_hard = self.get_Z_hard(self.log_alpha_agg, self.temperature, self.args)
		self.Z_combine_hard = self.get_Z_hard(self.log_alpha_combine, self.temperature, self.args)
		self.Z_comp_hard = self.get_Z_hard(self.log_alpha_comp, self.temperature, self.args)

		# temperature is initialted at message_passing-->load_searchspace()
		if epoch > 30 and epoch % 5 == 0 and self.temperature >= 1e-20:
			self.temperature *= 1e-1

		self.Z_hard_dict_micro['agg'].append(self.Z_agg_hard.cpu().tolist())
		self.Z_hard_dict_micro['combine'].append(self.Z_combine_hard.cpu().tolist())
		self.Z_hard_dict_micro['comp'].append(self.Z_comp_hard.cpu().tolist())
		self.Z_hard_dict_micro = dict(self.Z_hard_dict_micro)

	def derive_arch_micro(self, best_epoch):
		self.Z_agg_hard = self.Z_hard_dict_micro['agg'][best_epoch]
		self.Z_combine_hard = self.Z_hard_dict_micro['combine'][best_epoch]
		self.Z_comp_hard = self.Z_hard_dict_micro['comp'][best_epoch]


	def __init__(self, in_channels, out_channels, num_rels, num_bases, act=lambda x:x, cache=True, params=None):
		super(self.__class__, self).__init__(args=params)

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.num_bases 		= num_bases
		self.act 		= act
		# self.device		= None
		self.cache 		= cache			# Should be False for graph classification tasks

		# self.w_loop		= get_param((in_channels, out_channels));
		# self.w_in		= get_param((in_channels, out_channels));
		# self.w_out		= get_param((in_channels, out_channels));
		self.w_ent = get_param((in_channels, out_channels));

		self.rel_basis 		= get_param((self.num_bases, in_channels))
		self.rel_wt 		= get_param((self.num_rels*2, self.num_bases))
		self.w_rel 		= get_param((in_channels, out_channels))
		self.loop_rel 		= get_param((1, in_channels));

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)
		
		self.in_norm, self.out_norm = None, None
		self.in_index, self.out_index = None, None
		self.in_type, self.out_type = None, None
		self.loop_index, self.loop_type = None, None

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

		# self.combine_merger = nn.ModuleList([nn.Linear(3*out_channels, out_channels) for i in range(params.gcn_layer)])
		self.combine_merger = nn.Linear(3*out_channels, out_channels)
		self.Z_hard_dict_micro = ddict(list)
		self.init_alpha_micro()
		self.update_z_hard_micro(epoch=0)


	# def forward(self, x, edge_index, edge_type, rel_embed):
	def forward(self, emb_list_r, x, edge_index, edge_type, edge_norm=None, rel_embed=None, first_layer=False):
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.mm(self.rel_wt, self.rel_basis)
		if first_layer:
			emb_list_r.append(rel_embed)
			# print(emb_list_r[0].shape)
		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

		num_edges = edge_index.size(1) // 2
		num_ent   = x.size(0)

		if not self.cache or self.in_norm == None:
			self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
			self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

			self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
			self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

			self.in_norm     = self.compute_norm(self.in_index,  num_ent)
			self.out_norm    = self.compute_norm(self.out_index, num_ent)
		
		in_res		= self.propagate(self.Z_agg_hard, self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
		loop_res	= self.propagate(self.Z_agg_hard, self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
		out_res		= self.propagate(self.Z_agg_hard, self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		out = self.combine_trans(in_res, loop_res, out_res)

		if self.p.bias: out = out + self.bias
		# if self.b_norm: out = self.bn(out)
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]


	def combine_map(self, in_res, loop_res, out_res, combine):
		if combine == 'sum':
			x = in_res*(1/3) + loop_res*(1/3) + out_res*(1/3)
		if combine == 'concat':
			x = torch.cat([in_res, loop_res, out_res], axis=-1)
			x = self.combine_merger(x)
		return x

	def combine_trans(self, in_res, loop_res, out_res):
		y = []
		for combine in self.search_space['combine']:
			temp = self.combine_map(in_res, loop_res, out_res, combine)
			y.append(temp)
		out = torch.stack(y, dim=0)
		out = torch.einsum('ij,jkl -> ikl', self.Z_combine_hard, out).squeeze(0)
		return out


	def rel_transform_map(self, comp, ent_embed, rel_embed):
		if comp == 'corr':
			trans_embed = ccorr(ent_embed, rel_embed)
		elif comp == 'sub':
			trans_embed = ent_embed - rel_embed
		elif comp == 'mult':
			trans_embed = ent_embed * rel_embed
		else: raise NotImplementedError
		return trans_embed

	def rel_transform(self, ent_embed, rel_embed):
		comp_list = self.search_space['comp']
		y = []
		for comp in comp_list:
			temp = self.rel_transform_map(comp, ent_embed, rel_embed)
			y.append(temp)
		trans_embed = torch.stack(y, dim=0)
		trans_embed = torch.einsum('ij,jkl -> ikl', self.Z_comp_hard, trans_embed).squeeze(0)
		return trans_embed


	def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
		# weight 	= getattr(self, 'w_{}'.format(mode))
		weight = self.w_ent

		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		xj_rel  = self.rel_transform(x_j, rel_emb)
		out	= torch.mm(xj_rel, weight)

		if not self.p.no_edge_normalize:
			return out if edge_norm is None else out * edge_norm.view(-1, 1)
		else:
			return out


	def update(self, aggr_out):
		return aggr_out


	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges [Computing out-degree] [Should be equal to in-degree (undireted graph)]
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
