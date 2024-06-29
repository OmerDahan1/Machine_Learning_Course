import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    sum_of_squares = 0.0
    for label in np.unique(data[:,-1]):
        label_prob = np.mean(data[:,-1] == label)
        sum_of_squares += label_prob ** 2
    gini = 1 - sum_of_squares

    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    for label in np.unique(data[:,-1]):
        label_prob = np.mean(data[:,-1] == label)
        entropy -= label_prob * np.log2(label_prob)

    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        label_column = self.data[:,-1]
        unique_labels, counts = np.unique(label_column, return_counts=True)
        pred = unique_labels[np.argmax(counts)]

        return pred


    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)


    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """

        node_probability = self.data.shape[0] / n_total_sample
        goodness, _ = self.goodness_of_split(self.feature)
        self.feature_importance = node_probability * goodness



    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        current_impurity = self.impurity_func(self.data)
        instances = self.data.shape[0]
        feature_column = self.data[:, feature]
        feature_values = np.unique(self.data[:, feature])

        sum_impurity = 0

        for value in feature_values:
            count = np.sum(feature_column == value)
            data_trimmed = self.data[feature_column == value]
            groups[value] = data_trimmed
            impurity = self.impurity_func(data_trimmed)
            sum_impurity += ((count / instances) * impurity)

        goodness = current_impurity - sum_impurity  # gain

        if self.gain_ratio:
            split_information = 0
            for value in feature_values:
                count = np.sum(feature_column == value)
                instances = self.data.shape[0]
                count_log = np.log2(count / instances)
                sum_split_info = (count / instances) * count_log
                split_information -= sum_split_info

            if split_information != 0:
                goodness /= split_information
            else:
                goodness = 0

        return goodness, groups


    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        max_goodness = -1
        features = self.data.shape[1] - 1
        if self.depth == self.max_depth:
            self.terminal = True
            return
        for feature in range(features):
            goodness, groups = self.goodness_of_split(feature)
            if goodness > max_goodness:
                max_goodness = goodness
                self.feature = feature

        num_of_feature_values = len(np.unique(self.data[:,self.feature]))
        num_of_labels = len(np.unique(self.data[:,-1]))

        if max_goodness <= 0 or num_of_feature_values <= 1:
            self.terminal = True
            return

        X_square  = self.calc_chi()
        chi_found = False
        for key, sub_dict in chi_table.items():
            if self.chi in sub_dict:
                chi_found = True
                break
        if chi_found:
            if X_square <= chi_table[(num_of_feature_values - 1) * (num_of_labels - 1)][self.chi]:
                self.terminal = True
                return

        feature_column = self.data[:, self.feature]
        for value in np.unique(feature_column):
            child_data = self.data[feature_column == value]
            child = DecisionNode(data=child_data, impurity_func=self.impurity_func, depth=self.depth + 1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(child, value)

        if not self.children:
            self.terminal = True
            return

    def calc_chi(self):
        """
        Calculate the chi square value for the current node.

        Returns:
        - chi: the chi square value
        """
        chi = 0
        feature_column = self.data[:, self.feature]
        label_column = self.data[:,-1]
        for value in np.unique(feature_column):
            for label in np.unique(label_column):
                expected = np.sum(feature_column == value) * np.sum(label_column == label) / self.data.shape[0]
                observed = np.sum((feature_column == value) & (label_column == label))
                chi += (observed - expected) ** 2 / expected

        return chi


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func, chi=self.chi, max_depth=self.max_depth,
                                 gain_ratio=self.gain_ratio)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if not node.terminal:
                node.split()
                for child in node.children:
                    queue.append(child)

        return self.root

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        node = self.root
        found_child = True
        while not node.terminal and found_child:
            found_child = False
            for child in node.children:
                if (child.data[:, node.feature] == instance[node.feature]).all():
                    node = child
                    found_child = True
                    break
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """

        correct_predictions = 0
        total_instances = dataset.shape[0]

        for instance in dataset:
            prediction = self.predict(instance)
            actual_label = instance[-1]
            if prediction == actual_label:
                correct_predictions += 1

        accuracy = (correct_predictions / total_instances)
        return accuracy

    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, gain_ratio=True ,max_depth=max_depth)
        root = tree.build_tree()
        training_accuracy = tree.calc_accuracy(X_train)
        validation_accuracy = tree.calc_accuracy(X_validation)
        training.append(training_accuracy)
        validation.append(validation_accuracy)

    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for chi in chi_values:
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, gain_ratio=True, chi=chi)
        root = tree.build_tree()
        training_accuracy = tree.calc_accuracy(X_train)
        validation_accuracy = tree.calc_accuracy(X_test)
        chi_training_acc.append(training_accuracy)
        chi_validation_acc.append(validation_accuracy)
        max_depth = calc_depth(root)
        depth.append(max_depth)

    return chi_training_acc, chi_validation_acc, depth

def calc_depth(node):
    """
    Calculate the depth of a given tree

    Input:
    - node: a node in the decision tree.

    Output: the depth of the tree.
    """
    if node.terminal:
        return 0
    if not node:
        return 0
    depth = 0
    if node.children:
        for child in node.children:
            depth = max(depth, calc_depth(child))
    return depth + 1

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    if node.terminal:
        return 1
    if not node:
        return 0
    n_nodes = 1
    if node.children:
        for child in node.children:
            n_nodes += count_nodes(child)
    return n_nodes






