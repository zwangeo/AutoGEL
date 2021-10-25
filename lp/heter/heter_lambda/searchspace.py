from collections import defaultdict as ddict


class SearchSpace(object):
    def __init__(self):
        self.agg = ['sum', 'mean', 'max']
        self.combine = ['sum', 'concat']
        self.comp = ['sub', 'mult', 'corr']
        # self.layer_connect = ['stack', 'skip_sum', 'skip_cat']
        # self.layer_agg = ['none', 'concat', 'max_pooling']
        self.layer_agg = ['stack', 'sum', 'concat', 'max_pooling']

        self.search_space = {'agg': self.agg,
                             'combine': self.combine,
                             'comp': self.comp,
                             'layer_agg': self.layer_agg}

        self.dims = list(self.search_space.keys())
        self.choices = []
        self.num_choices = {}
        for dim in self.dims:
            self.choices.append(self.search_space[dim])
            self.num_choices[dim] = len(self.search_space[dim])

    def get_search_space(self):
        return self.search_space