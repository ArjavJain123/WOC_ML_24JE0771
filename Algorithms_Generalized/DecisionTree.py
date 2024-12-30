import numpy as np
import pandas as pd
class Node:
  #The * is used to seperate make the value as a keyword argument instead of positional argument. So now, when passing in "value",
  # I have to explicitly write value = 5 (or any number)
  def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value
  def is_leaf_node(self):
    return self.value is not None

class DecisionTree:
  def __init__(self, min_samples_split=2, max_depth =150, n_features = None):
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.n_features = n_features
    self.root = None
  def fit(self, X, y):
    self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
    #The above line is eq to saying that, if n_features is not defined then set it to X.shape[0]. if defined then set it to min of X.shape[0] and defined value
    self.root = self._grow_tree(X, y)

  #in below, the role of the underscore is that, this function is meant to be kept private, and only for use within the class
  # though, any1 from outside can still access it by calling this class method.
  def _grow_tree(self, X, y, depth=0):
    n_samples, n_feats = X.shape
    n_labels = len(np.unique(y)) #np.unique(y) gives a array that removes all duplicates from y and contains only unique lables
    #then, n_labels = len(np.unique(y)) just means the total number of unique values in y
    #if n_labels is 1, then it is surely a leaf node and the value of the leaf node will be the value of any element in y, since all are same

    #stopping conditions:
    if(depth>= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
      # print("Max depth reached!")
      leaf_value = self._most_common_label(y)
      return Node(value=leaf_value)

    feat_idxs = np.random.choice(n_feats, self.n_features, replace = False)

    #find best split:
    best_feature, best_thresh = self._best_split(X, y, feat_idxs)

    #create children:
    left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
    left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
    right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
    return Node(best_feature, best_thresh, left, right)

  def _most_common_label(self, y):
    unique_values, counts = np.unique(y, return_counts=True)
    most_common_index = np.argmax(counts)
    return unique_values[most_common_index]

  def _best_split(self, X, y, feat_idxs):
    best_gain = -1
    split_idx, split_threshold= None, None
    for feat_idx in feat_idxs:
      X_column = X[:, feat_idx]
      thresholds = np.unique(X_column)
      for thr in thresholds:
        gain = self._information_gain(y,X_column,thr)
        if gain > best_gain:
          best_gain = gain
          split_idx = feat_idx
          split_threshold = thr
    return split_idx, split_threshold

  def _information_gain(self, y, X_column, threshold):
    #ccalculte parent entropy:
    parent_entropy = self._entropy(y)
    #create children
    left_idxs, right_idxs = self._split(X_column, threshold)
    if len(left_idxs) == 0 or len(right_idxs) == 0:
      return 0

    #calc weighted avg:
    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
    child_entropy = (n_l*e_l) + (n_r*e_r)
    child_entropy = child_entropy / n
    #finally, calc ig
    information_gain = parent_entropy - child_entropy
    return information_gain

  def _split(self, X_column, split_thresh):
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    return left_idxs, right_idxs
  def _entropy(self, y):
    unique_values, counts = np.unique(y, return_counts=True)
    entropy = 0.0
    count_sum = np.sum(counts)
    for count in counts:
      proba = count/count_sum
      entropy += proba*np.log2(proba) if proba > 0 else 0
    return (-1*entropy)
  def predict(self, X):
    return np.array([self._traverse(x, self.root) for x in X])

  def _traverse(self, x, node):
    if node.is_leaf_node():
      return node.value
    if x[node.feature] <= node.threshold:
      return self._traverse(x, node.left)
    return self._traverse(x, node.right)


  def get_conf_mat(self, Y_act, Y_pred):
    n_cls = len(np.unique(Y_act))
    con_mat = np.zeros((n_cls, n_cls))
    for indx, label in enumerate(Y_act):
      con_mat[label, Y_pred[indx]] += 1
    
    return con_mat
  
  def print_con_mat(self, con_mat):
    print("Confusion matrix with predictions on X axis and actual values on Y")
    classes= np.arange(con_mat.shape[0])
    table = pd.DataFrame(con_mat, index=classes, columns=classes)
    print(table)
    
  def get_precision(self,con_mat):
    precs = np.zeros((con_mat.shape[0],))
    for i in range(con_mat.shape[0]):
      precs[i] = con_mat[i,i]/np.sum(con_mat[:,i])
      
    return precs
  
  def get_recall(self, con_mat):
    recs = np.zeros((con_mat.shape[0],))
    for i in range(con_mat.shape[0]):
      recs[i] = con_mat[i,i]/np.sum(con_mat[i,:])
      
    return recs
    
  def get_f1s(self, con_mat):
    precs = self.get_precision(con_mat)
    recs = self.get_recall(con_mat)
    f1s = 2*recs*precs/(recs + precs)
    return f1s
  
  def classification_report(self, Y_act, Y_pred):
    con_mat = self.get_conf_mat(Y_act, Y_pred)
    precs = self.get_precision(con_mat)
    recs = self.get_recall(con_mat)
    f1s = self.get_f1s(con_mat)
    supports = con_mat.sum(axis=1)
    classes= np.arange(con_mat.shape[0])
    reports = []
    for i in range(con_mat.shape[0]):
      reports.append([precs[i], recs[i], f1s[i], supports[i]])
    table1 = pd.DataFrame(reports,index=classes, columns=['precision','recall','f1-score','support'])
    # print(table1)
    wted_prec = np.average(precs, weights=supports)
    wted_rec = np.average(recs, weights=supports)
    wted_f1 = np.average(f1s, weights=supports)
    tot_support = np.sum(supports)
    wted_data = [[wted_prec,wted_rec,wted_f1,tot_support]]
    table2 = pd.DataFrame(wted_data,index=['Average (weighted)'],columns=['precision', 'recall', 'f1-score', 'support'])
    # print('\n\n', table2)
    table = pd.concat([table1, table2])
    print(table)
