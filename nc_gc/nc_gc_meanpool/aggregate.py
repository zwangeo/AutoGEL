from torch_geometric.nn import MessagePassing


class Sum_AGG(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Sum_AGG, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        neighbor_info = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return neighbor_info

    def massage(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class Mean_AGG(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Mean_AGG, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        neighbor_info = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return neighbor_info

    def massage(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class Max_AGG(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Max_AGG, self).__init__(aggr='max')

    def forward(self, x, edge_index):
        neighbor_info = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return neighbor_info

    def massage(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out