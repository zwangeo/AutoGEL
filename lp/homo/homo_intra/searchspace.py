class SearchSpace(object):
    def __init__(self):
        # self.gcnconv = ['GINConv', 'GCNConv', 'SAGEConv', 'GATConv']
        self.agg = ['sum', 'mean', 'max']
        self.combine = ['sum', 'concat']
        self.act = ['relu', 'prelu']
        self.layer_connect = ['stack', 'skip_sum', 'skip_cat']
        self.layer_agg = ['none', 'concat', 'max_pooling']
        # self.pool = ['sum', 'diff', 'hadamard', 'max', 'concat']
        # self.pool = ['sum']
        self.pool = ['sum', 'max', 'concat']

        self.search_space = {'agg': self.agg,
                             'combine': self.combine,
                             'act': self.act,
                             'layer_connect': self.layer_connect,
                             'layer_agg': self.layer_agg,
                             'pool': self.pool}

        self.dims = list(self.search_space.keys())
        self.choices = []
        self.num_choices = {}
        for dim in self.dims:
            self.choices.append(self.search_space[dim])
            self.num_choices[dim] = len(self.search_space[dim])

    def get_search_space(self):
        return self.search_space