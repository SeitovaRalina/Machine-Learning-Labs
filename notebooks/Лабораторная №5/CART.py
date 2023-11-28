import numpy as np

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # for decision node - узел принятия решений, разделяется
        self.feature_index = feature_index
        self.threshold = threshold # пороговое значение
        self.left = left
        self.right = right
        self.info_gain = info_gain # прирост инфы
        
        # for leaf node - конечный узел, больше не разделяется
        self.value = value

class ClassificationAndRegressionTrees():
    def __init__(self, min_samples_split=2, max_depth=2, criterion = None):
        # initialize the root of the tree - корень дерева
        self.root = None

        # classification or regression??? - функция для расчета примеси
        self.criterion = criterion
        self.functions = {'entropy': self.entropy, 
                     'gini' : self.gini_index,
                     'squared_error': self.squared_error,
                     'absolute_error': self.absolute_error}
        
        # stopping conditions - условия остановки
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    # рекурсивное построение дерева    
    def build_tree(self, dataset, curr_depth=0): 
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X) # число образцов = число строк, число признаков = число столбцов
        
        # разделять до тех пор, пока не будут выполнены условия остановки
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # находим лучшее разделение
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # строить дерево, пока есть прирост инфы (> 0), иначе это уже будет листовой узел
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # вернуть decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # вычислить leaf node
        leaf_value = self.find_leaf_value(Y, self.criterion)

        # вернуть leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features): # для одного уровня дерева
        # словарь для хранения лучшего разделения
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            # столбец текущего признака
            feature_values = dataset[:, feature_index]
            # пройтись по всем уникальным значениям признака, т.е. предполагаемым пороговым значениям
            for threshold in np.unique(feature_values):
                # создать разделение по текущему признаку и текущему порогу
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # если ветки потомков не пусты, то продолжаем
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # вычислить информационный прирост
                    curr_info_gain = self.information_gain(y, left_y, right_y, self.criterion)
                    # обновить лучшее разделение
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        # левая ветвь - выбрать те строки, которые в преполагаемом столбце имеют значение не меньше порогового
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        # правая ветвь - -||- меньше порогового
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def find_leaf_value(self, Y, criterion):
        # если задача классификации
        if criterion in ['entropy', 'gini']:
            # задача классификации
            # мюсли : не факт, что в конечный узел войдет только один класс, поэтому берем класс с наибольшим количеством образцов
            list_Y = list(Y)
            leaf_value = max(list_Y, key=list_Y.count)
        else:
            # задача регрессии
            leaf_value = np.mean(Y)
        return leaf_value

    def information_gain(self, parent, l_child, r_child, criterion):      
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.functions[criterion](parent) - (weight_l*self.functions[criterion](l_child) + weight_r*self.functions[criterion](r_child))
        return gain
    
    def entropy(self, y):
        # все признаки, которые вошли в ветвь потомка
        class_labels = np.unique(y)
        entropy = 0
        for label in class_labels:
            # расчет вероятности класса label
            p_cls = len(y[y == label]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        # все признаки, которые вошли в ветвь потомка
        class_labels = np.unique(y)
        return 1 - sum((len(y[y == label]) / len(y))**2 for label in class_labels)
    
    def squared_error(self, y):
        class_labels = np.unique(y)
        f = lambda x: pow(x - class_labels.mean(), 2)
        return sum(map(f, class_labels))/len(class_labels)
    
    def absolute_error(self, y):
        class_labels = np.unique(y)
        f = lambda x: np.abs(x)
        return sum(map(f, class_labels))/len(class_labels)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in np.array(X)]
        return predictions
    
    def make_prediction(self, x, tree):
        if tree.value!=None: 
            return tree.value
        feature_val = x[tree.feature_index]

        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def print_tree(self, tree=None, indent="|   "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print('%.4f'%tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", '%.3f'%tree.threshold)
            print(f"{indent}left: ", end="")
            self.print_tree(tree.left, indent + '|   ')
            print(f"{indent}right: ", end="")
            self.print_tree(tree.right, indent + '|   ')