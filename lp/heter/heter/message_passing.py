from collections import defaultdict as ddict
import inspect
from torch_scatter import scatter
import torch
import torch.nn.functional as F
import torch.nn as nn
from searchspace import *


def scatter_(z_agg_hard, agg_list, src, index, dim_size=None):
	r"""Aggregates all values from the :attr:`src` tensor at the indices
	specified in the :attr:`index` tensor along the first dimension.
	If multiple indices reference the same location, their contributions
	are aggregated according to :attr:`name` (either :obj:`"add"`,
	:obj:`"mean"` or :obj:`"max"`).

	Args:
		name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
			:obj:`"max"`).
		src (Tensor): The source tensor.
		index (LongTensor): The indices of elements to scatter.
		dim_size (int, optional): Automatically create output tensor with size
			:attr:`dim_size` in the first dimension. If set to :attr:`None`, a
			minimal sized output tensor is returned. (default: :obj:`None`)

	:rtype: :class:`Tensor`
	"""
	# if name == 'add': name = 'sum'
	# assert name in ['sum', 'mean', 'max']

	# agg_list = ['sum', 'mean', 'max']
	y = []
	for agg in agg_list:
		temp = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=agg)
		y.append(temp)
	out = torch.stack(y, dim=0)
	out = torch.einsum('ij,jkl -> ikl', z_agg_hard, out).squeeze(0)
	return out[0] if isinstance(out, tuple) else out


class MessagePassing(torch.nn.Module):
	r"""Base class for creating message passing layers

	.. math::
		\mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
		\square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
		\left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

	where :math:`\square` denotes a differentiable, permutation invariant
	function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
	and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
	MLPs.
	See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
	create_gnn.html>`__ for the accompanying tutorial.

	"""

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


	def __init__(self, args=None):
		super(MessagePassing, self).__init__()
		self.device = self.get_device(args)
		self.args = args
		self.layers = args.gcn_layer

		self.message_args = inspect.getargspec(self.message)[0][1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
		self.update_args  = inspect.getargspec(self.update)[0][2:]	# Same for update function starting from 3rd argument | first=self, second=out

		# self.Z_hard_dict = ddict(list)
		self.load_searchspace()
		# self.init_alpha()
		# self.update_z_hard()


	def propagate(self, z_agg_hard, edge_index, **kwargs):
		r"""The initial call to start propagating messages.
		Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
		:obj:`"max"`), the edge indices, and all additional data which is
		needed to construct messages and to update node embeddings."""

		# assert aggr in ['add', 'mean', 'max']
		# agg_list = self.search_space['agg']
		kwargs['edge_index'] = edge_index


		size = None
		message_args = []
		for arg in self.message_args:
			if arg[-2:] == '_i':					# If arguments ends with _i then include indic
				tmp  = kwargs[arg[:-2]]				# Take the front part of the variable | Mostly it will be 'x', 
				size = tmp.size(0)
				message_args.append(tmp[edge_index[0]])		# Lookup for head entities in edges
			elif arg[-2:] == '_j':
				tmp  = kwargs[arg[:-2]]				# tmp = kwargs['x']
				size = tmp.size(0)
				message_args.append(tmp[edge_index[1]])		# Lookup for tail entities in edges
			else:
				message_args.append(kwargs[arg])		# Take things from kwargs

		update_args = [kwargs[arg] for arg in self.update_args]		# Take update args from kwargs

		out = self.message(*message_args)
		# out = scatter_(aggr, out, edge_index[0], dim_size=size)		# Aggregated neighbors for each vertex
		# out = scatter_(self.Z_agg_hard.view(1,-1), self.search_space['agg'], out, edge_index[0], dim_size=size)
		out = scatter_(z_agg_hard, self.search_space['agg'], out, edge_index[0], dim_size=size)
		out = self.update(out, *update_args)

		return out

	def message(self, x_j):  # pragma: no cover
		r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
		for each edge in :math:`(i,j) \in \mathcal{E}`.
		Can take any argument which was initially passed to :meth:`propagate`.
		In addition, features can be lifted to the source node :math:`i` and
		target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
		variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

		return x_j

	def update(self, aggr_out):  # pragma: no cover
		r"""Updates node embeddings in analogy to
		:math:`\gamma_{\mathbf{\Theta}}` for each node
		:math:`i \in \mathcal{V}`.
		Takes in the output of aggregation as first argument and any argument
		which was initially passed to :meth:`propagate`."""

		return aggr_out
