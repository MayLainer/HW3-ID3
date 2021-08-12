import numpy
import numpy as np
import math
from random import randrange
import pandas as pd
from numpy import log2 as log


# changes:
# if more then 0.3 are sick, we label the node as sick


def get_random_label(num_of_sick, num_of_healthy):
    random_num = randrange(8)
    if num_of_sick == 1:
        return 0
    if 1 < num_of_sick < 7:
        if random_num < num_of_sick:
            return 1
    else:
        return 0


class PersonalizedID3:
    def __init__(self):
        self.pruning_number = 8

    # train and test are numpy.ndarray
    def fit_predict(self, train, test):
        decision_tree = Tree(train)
        self.make_decision_tree_with_pruning(decision_tree.root)

        return self.get_label(decision_tree, test)

    def get_label(self, decision_tree, test):
        result_array = []

        for i in range(test.shape[0]):
            result_array.append(self.result_from_tree(decision_tree.root, test[i]))
        f_result = np.array(result_array)
        return f_result

    def result_from_tree(self, curr_node, test_row):
        if curr_node.is_leaf is True:
            return curr_node.label

        func_result = curr_node.node_feature.run_feature(test_row)
        if func_result == 0:
            return self.result_from_tree(curr_node.zero_child, test_row)
        else:
            return self.result_from_tree(curr_node.one_child, test_row)

    def make_decision_tree_with_pruning(self, curr_node):
        curr_training_set = curr_node.training_set
        ndarray_size = curr_training_set.shape
        if ndarray_size[0] == 0:
            curr_node.is_leaf = True
            return

        if curr_node.num_of_sick == ndarray_size[0] or \
            curr_node.num_of_healthy == ndarray_size[0]:
            curr_node.is_leaf = True
            return

        if ndarray_size[0] < self.pruning_number:
            curr_node.label = get_random_label(curr_node.num_of_sick, curr_node.num_of_healthy)
            curr_node.is_leaf = True
            return

        # calc the entropy for each feature, then pick the max one
        best_feature = self.select_feature(curr_node)
        curr_node.node_feature = best_feature
        self.split_and_make_children(curr_node, best_feature)

        self.make_decision_tree_with_pruning(curr_node.zero_child)
        self.make_decision_tree_with_pruning(curr_node.one_child)

    def split_and_make_children(self, curr_node, feature):
        training_set = curr_node.training_set
        group_zero = []
        group_one = []
        for i in range(training_set.shape[0]):
            feature_result = feature.run_feature(training_set[i])
            if feature_result == 0:
                group_zero.append(training_set[i])
            else:
                group_one.append(training_set[i])

        ndarray_zero = np.array(group_zero)
        ndarray_one = np.array(group_one)
        node_zero = Node(ndarray_zero, -1)
        node_one = Node(ndarray_one, -1)
        curr_node.add_child(node_zero, 0)
        curr_node.add_child(node_one, 1)
        return

    def select_feature(self, curr_node):
        training_set = curr_node.training_set
        best_feature = None
        max_ig = -math.inf
        for i in range(training_set[0].size):
            if i == 0:
                continue
            curr_feature, curr_gain = self.find_best_feature_by_num(curr_node, training_set, i)
            if curr_gain >= max_ig:
                max_ig = curr_gain
                best_feature = curr_feature
        return best_feature

    def find_best_feature_by_num(self, curr_node, curr_training_set, feature_num):
        copy_ndarray = np.copy(curr_training_set)
        #np.sort(sorted_training_set, axis=1)
        sorted_training_set = copy_ndarray[numpy.argsort(copy_ndarray[:, feature_num])]
        #sorted_training_set[sorted_training_set[:feature_num].argsort()]
        parent_entropy = self.calc_entropy(curr_node)
        max_feature = None
        max_gain = -math.inf
        ndarray_size = curr_training_set.shape
        for i in range(ndarray_size[0] - 1):
            t = 0.5 * (sorted_training_set[i][feature_num] + sorted_training_set[i + 1][feature_num])
            new_feature = Feature(feature_num, t)
            curr_ig = self.get_ig(new_feature, sorted_training_set, parent_entropy)
            if curr_ig >= max_gain:
                max_feature = new_feature
                max_gain = curr_ig

        return max_feature, max_gain

    def get_ig(self, feature, sorted_training_set, parent_entropy):
        group_zero = []
        group_one = []
        ndarray_size = sorted_training_set.shape
        for i in range(ndarray_size[0]):
            feature_result = feature.run_feature(sorted_training_set[i])
            if feature_result == 0:
                group_zero.append(sorted_training_set[i])
            else:
                group_one.append(sorted_training_set[i])
        ndarray_zero = np.array(group_zero)
        ndarray_one = np.array(group_one)
        node_zero = Node(ndarray_zero, -1)
        node_one = Node(ndarray_one, -1)
        ig = parent_entropy
        if len(group_zero) != 0:
            ig -= (ndarray_zero.shape[0] / ndarray_size[0]) * self.calc_entropy(node_zero)
        if len(group_one) != 0:
            ig -= (ndarray_one.shape[0] / ndarray_size[0]) * self.calc_entropy(node_one)
        return ig

    def calc_entropy(self, curr_node):
        entropy = 0
        training_set = curr_node.training_set
        ndarray_size = training_set.shape
        prob_sick = curr_node.num_of_sick / ndarray_size[0]
        prob_healthy = curr_node.num_of_healthy / ndarray_size[0]
        # for i in range(ndarray_size[0]):
        #     if training_set[i][0] == 'M':
        #         entropy -= prob_sick * log(prob_sick)
        #     else:
        #         entropy -= prob_healthy * log(prob_healthy)
        if prob_sick != 0:
            entropy -= prob_sick * log(prob_sick)
        if prob_healthy != 0:
            entropy -= prob_healthy * log(prob_healthy)
        return entropy


class Node:
    def __init__(self, training_set, parent_label):
        self.zero_child = None  # func return zero to those children
        self.one_child = None
        self.training_set = training_set
        ndarray_size = training_set.shape
        self.is_leaf = False

        healthy_num, sick_num = 0, 0
        if training_set.size == 0:
            self.label = parent_label
            self.is_leaf = True
        else:
            for i in range(ndarray_size[0]):
                if training_set[i][0] == 'M':  # check if M is healthy or sick
                    sick_num += 1
                else:
                    healthy_num += 1
            if sick_num > healthy_num:  # if all of the examples has the same label, we make a leaf
                self.label = 1
            else:
                self.label = 0

        self.parent_label = parent_label
        self.num_of_sick = sick_num
        self.num_of_healthy = healthy_num
        self.node_feature = None

    def add_child(self, new_node, index):
        new_node.parent_label = self.label
        if index == 0:
            self.zero_child = new_node
        else:
            self.one_child = new_node

    def set_feature(self, feature):
        self.node_feature = feature


class Tree:
    def __init__(self, training_set):
        self.root = Node(training_set, -1)


class Feature:
    def __init__(self, feature_num, t):
        self.feature_num = feature_num
        self.t = t

    def run_feature(self, row):
        if row[self.feature_num] < self.t:
            return 0
        return 1
