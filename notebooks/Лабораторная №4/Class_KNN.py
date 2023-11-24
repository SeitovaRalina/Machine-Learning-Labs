import numpy as np

class K_Nearest_Neighbors:
    def __init__(self, n_neighbors=5, p=2, metric='minkowski'):
        self.k = n_neighbors
        if(metric=='minkowski'):
            self.func = lambda X,Y: np.sqrt(np.sum(list(map(lambda x,y: np.abs(x-y)**p,X,Y))))
        elif(metric=='euclidean'):
            self.func = lambda X,Y:  np.sqrt(np.sum(list(map(lambda x,y: (x-y)**2,X,Y))))
        elif(metric=='manhattan'):
            self.func = lambda X,Y: np.sqrt(np.sum(list(map(lambda x,y: np.abs(x-y),X,Y))))
        else:
            raise Exception() 
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    def predict(self, X_test):
        y_pred = []
        for j in range(len(X_test)):
            # get neighbors
            distances = [(self.y_train[i], self.func(self.X_train[i], X_test[j])) for i in range(len(self.X_train + 1))]
            distances.sort(key=lambda elem: elem[1])
            neighbors = [distances[i][0] for i in range(self.k)]

            # prediction
            count = {}
            for instance in neighbors:
                if instance in count:
                    count[instance] +=1
                else :
                    count[instance] = 1
            target = max(count.items(), key=lambda x: x[1])[0] # метка с наибольшим количеством встречаемости среди k соседей
            y_pred.append(target)
        return np.array(y_pred)