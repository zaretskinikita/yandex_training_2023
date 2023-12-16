import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            self.indices_list.append(np.random.choice(range(len(data)), size=len(data)))
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            indices = self.indices_list[bag]
            data_bag, target_bag = data[indices], target[indices]
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        prediction = np.array([0.0] * len(data))
        for model in self.models_list:
            prediction += model.predict(data)
        return prediction / len(self.models_list)
            
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        for i in range(len(self.data)):
            preds = []
            for bag in range(self.num_bags):
                if i not in self.indices_list[bag]:
                    preds.append(float(self.models_list[bag].predict(self.data[i].reshape(1, -1))))
            list_of_predictions_lists[i] = preds
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        avg_preds = []
        for i in range(len(self.list_of_predictions_lists)):
            if len(self.list_of_predictions_lists[i]) > 0:
                avg_preds.append(np.mean(self.list_of_predictions_lists[i]))
            else:
                avg_preds.append(None)
        self.oob_predictions = np.array(avg_preds)
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        oob_preds = self.oob_predictions[self.oob_predictions != np.array(None)]
        target = self.target[self.oob_predictions != np.array(None)]
        return np.mean((oob_preds - target)**2)