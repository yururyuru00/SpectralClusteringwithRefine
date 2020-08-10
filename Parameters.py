class P:
    map_dataset_type = {'football' : 'social_net',
                                     'politicsuk' : 'social_net',
                                     'olympics' : 'social_net',
                                     'cora' : 'citation_net',
                                     'citeseer' : 'citation_net'}

    @classmethod
    def set(cls, settings):
        #dataset and number of clusters
        cls.dataset = settings.dataset
        cls.dataset_type = cls.map_dataset_type[cls.dataset]
        cls.cluster_size = settings.c
        
        #parameters for our parameters
        cls.sigma = settings.sigma
        cls.theta = settings.theta
        cls.delta = settings.delta
        cls.number_of_updates = settings.m
        cls.disable_normalization = settings.disable

        #parameters for SClump (tuned by SClump Algorithm)
        cls.alpha = 1.
        cls.beta = 6.
        cls.gamma = 0.5