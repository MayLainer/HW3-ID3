import numpy
import numpy as np
import math
import pandas as pd
from numpy import log2 as log
import sklearn.model_selection
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt


def calc_precision(temp_test, results):
    diff = 0
    same = 0
    for j in range(results.size):
        if results[j] == 0 and temp_test[j][0] == 'B':
            same += 1
        elif results[j] == 1 and temp_test[j][0] == 'M':
            same += 1
        else:
            diff += 1
    return same


def plot_distance_and_expanded_wrt_weight_figure(
        problem_name: str,
        pruning_number,
        precision_number):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    TODO [Ex.20]: Complete the implementation of this method.
    """

    pruning_number, precision_number = np.array(pruning_number), np.array(precision_number)
    assert len(pruning_number) == len(precision_number)
    assert len(pruning_number) > 0
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(pruning_number)

    fig, ax1 = plt.subplots()

    # TODO: Plot the total distances with ax1. Use `ax1.plot(...)`.
    # TODO: Make this curve colored blue with solid line style.
    # TODO: Set its label to be 'Solution cost'.
    # See documentation here:
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also Google for additional examples.
    # raise NotImplementedError  # TODO: remove this line!

    p1, = ax1.plot(pruning_number, precision_number, 'b-',
                   label="Plot")  # TODO: pass the relevant params instead of `...`.

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('precision number', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('pruning number')

    # Create another axis for the #expanded curve.

    # TODO: Plot the total expanded with ax2. Use `ax2.plot(...)`.
    # TODO: Make this curve colored red with solid line style.
    # TODO: Set its label to be '#Expanded states'.
    # raise NotImplementedError  # TODO: remove this line!
    curves = [p1]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    fig.tight_layout()
    plt.title(f'{problem_name}')
    plt.show()


def experiment(df_in):
    # df = pd.read_csv('train.csv', sep=',', header=None)  ---- old
    df = df_in
    testing_set = df.to_numpy()

    training_set = np.array(df)
    result_array = []
    id3 = ID3()
    prunning_number, precision_number = [], []
    max_per = -math.inf
    max_k = -math.inf

    kf = KFold(n_splits=5, shuffle=True, random_state=206719163)
    """ we run the experiment with 35 values, in order to make the experiment shorter,
        you should change the number to a smaller one"""
    for k in range(35):
        for train_index, test_index in kf.split(training_set):
            temp_test = []
            temp_train = []

            for i in range(testing_set.shape[0]):
                if i in train_index:
                    temp_train.append(training_set[i])
                else:
                    temp_test.append(training_set[i])
            cut_training_set = np.array(temp_train)
            cut_testing_set = np.array(temp_test)
            results = id3.experiment_id3(cut_training_set, cut_testing_set, k)
            result_array.append(calc_precision(temp_test, results))

        per_num = (sum(result_array) / 301)
        precision_number.append(per_num)  # precision number
        prunning_number.append(k)
        if per_num >= max_per:
            max_per = per_num
            max_k = k
        result_array = []

    # --------------------------------------------------------------------------
    # remove the comment below to print the graph
    """plot_distance_and_expanded_wrt_weight_figure("precision in result of pruning number", prunning_number,
                                                 precision_number)
    """

class ID3:
    def __init__(self):
        self.pruning_number = 0

    # train and test are numpy.ndarray
    def fit_predict(self, train, test):
        decision_tree = Tree(train)
        self.make_decision_tree(decision_tree.root)

        return self.get_label(decision_tree, test)

    # Experiment_id3:
    # Input:    This method gets a training set and testing set (both: numpy.ndarray) and pr_number-
    #           the min number of examples in a node that is allowed in order to split the mode
    # Output:   This method returns a result for each line in the test set (numpy.ndarray)
    def experiment_id3(self, train, test, pr_number):
        self.pruning_number = pr_number
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

    def make_decision_tree(self, curr_node):
        curr_training_set = curr_node.training_set
        ndarray_size = curr_training_set.shape
        if ndarray_size[0] == 0:
            curr_node.is_leaf = True
            return

        if curr_node.num_of_sick == ndarray_size[0] or \
                curr_node.num_of_healthy == ndarray_size[0]:
            curr_node.is_leaf = True
            return

        # calc the entropy for each feature, then pick the max one
        best_feature = self.select_feature(curr_node)  # two functions or one?
        curr_node.node_feature = best_feature
        self.split_and_make_children(curr_node, best_feature)

        self.make_decision_tree(curr_node.zero_child)
        self.make_decision_tree(curr_node.one_child)

    def make_decision_tree_with_pruning(self, curr_node):
        curr_training_set = curr_node.training_set
        ndarray_size = curr_training_set.shape
        if ndarray_size[0] == 0:
            curr_node.is_leaf = True
            return

        if ndarray_size[0] < self.pruning_number:
            curr_node.is_leaf = True
            return

        if curr_node.num_of_sick == ndarray_size[0] or \
            curr_node.num_of_healthy == ndarray_size[0]:
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
