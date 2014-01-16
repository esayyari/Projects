class Transformer:
    def __init__(self,**kwrds):
        self.set_params(**kwrds)
        
    def read_data(self, data_file_name):
        """
        read_data(data_file_name) -> [y, x]

        Read LIBSVM-format data from data_file_name and return labels y
        and data instances x.
        """
        prob_y = []
        prob_x = []
        for line in open(data_file_name):
                line = line.split(None, 1)
                # In case an instance with all zero features
                if len(line) == 1: line += ['']
                label, features = line
                xi = {}
                for e in features.split():
                        ind, val = e.split(":")
                        xi[int(ind)] = float(val)
                prob_y += [float(label)]
                prob_x += [xi]
        return (prob_y, prob_x)


    def fit_transform(self,X,y=None,**kwrds):
        self.fit(X,y,**kwrds)
        return self.transform(X)

    def fit(self,X,y=None,**kwrds):
        self.set_params(**kwrds)

    def set_params(self,**kwrds):
        pass
    
    def get_params(self,deep=False):
        return dict()
    
    def transform(self,X):
        return None
