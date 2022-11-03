from joblib import dump, load

from sklearn.tree import _tree
import numpy as np


class MyNode():
  """
  A class used to represent a Node of a tree

  ...

  Attributes
  ----------
  node_name : str
      The name of the node, for identification
  state : str
      In case of a leaf node, its predicted class
  left_node : MyNode
      In case of a decision node, a reference to the left node
  right_node : MyNode
      In case of a decision node, a reference to the right node
  threshold_name : str
      In case of a decision node, the name of the feature/condition in it
  threshold_value : float
      In case of a decision node, the value for the condition cutoff in it

  """
  def __init__(self, node_name, state, left_node, right_node, threshold_name, threshold_value):
    self.node_name = node_name
    self.state = state
    self.left_node = left_node
    self.rigth_node = right_node
    self.threshold_name = threshold_name
    self.threshold_value = threshold_value

  def __str__(self):
    return f"Node[ ({self.node_name}), state:{self.state}, threshold_name: {self.threshold_name}, , threshold_value: {self.threshold_value} ]"
  def __repr__(self):
      return str(self)

def _tree_to_tree(tree, feature_names):
  """Prunnes a given DecisionTreeClassifier, removing redundant conditions.

  Parameters
  ----------
  tree : DecisionTreeClassifier
      The model intended to be prunned
  feature_names : list[str]
      List containing the names of the features of the tree, in order

  Returns
  ----------
  (int, int)
      A tuple containing the number of decision nodes and the number of leaf 
      nodes, respectively

  """
  tree_ = tree.tree_
  feature_name = [              
      "_".join(feature_names[i].split()) if i != _tree.TREE_UNDEFINED else "undefined!"
      for i in tree_.feature
  ]

  def recurse(node, depth, node_name):

      actual_node = MyNode(node_name, None, None, None, None, None)

      if tree_.feature[node] != _tree.TREE_UNDEFINED:
          name = feature_name[node]
          threshold = tree_.threshold[node]

          actual_node.threshold_name = name
          actual_node.threshold_value = threshold
          
          left_c = recurse(tree_.children_left[node], depth + 1, node_name + "left")
          right_c = recurse(tree_.children_right[node], depth + 1, node_name + "right")

          if left_c.state == right_c.state and left_c.state:
            return left_c

          actual_node.left_node = left_c
          actual_node.right_node = right_c

          return actual_node

      else:
          actual_node.state = tree.classes_[np.argmax(tree_.value[node][0])]
          return actual_node

  return recurse(0, 1, "root")


def _prunned_tree_to_code_in_python(tree, feature_names):
  """Prints a prunned DecisionTreeClassifier, in Python3 code.

  Parameters
  ----------
  tree : MyNode
      Root node of the tree
  feature_names : list[str]
      List containing the names of the features of the tree, in order

  Returns
  ----------
  (int, int)
      A tuple containing the number of decision nodes and the number of leaf 
      nodes, respectively

  """
  print("def tree({}):".format(", ".join(["_".join(i.split()) for i in feature_names])) )

  if_counts = 0
  total_leafs = 0

  def recurse(node, depth):
      indent = "  " * depth

      nonlocal if_counts
      nonlocal total_leafs

      if node.state == None:
          name = node.threshold_name
          threshold = node.threshold_value
          if_counts += 1
          print("{}if {} <= {}:".format(indent, name, threshold))
          recurse(node.left_node, depth + 1)
          print("{}else:  # if {} > {}".format(indent, name, threshold) )
          recurse(node.right_node, depth + 1)
      else:
          total_leafs += 1
          print("{}return '{}'".format(indent, node.state))


  recurse(tree, 1)
  return if_counts, total_leafs

def _prunned_tree_to_code_in_c(tree, feature_names):
  """Prints a prunned DecisionTreeClassifier, in C code.

  Parameters
  ----------
  tree : MyNode
      Root node of the tree
  feature_names : list[str]
      List containing the names of the features of the tree, in order

  Returns
  ----------
  (int, int)
      A tuple containing the number of decision nodes and the number of leaf 
      nodes, respectively

  """
  print("char tree({}){{".format(", ".join(["_".join(i.split()) for i in feature_names])) )

  if_counts = 0
  total_leafs = 0

  def recurse(node, depth):
      indent = "  " * depth

      nonlocal if_counts
      nonlocal total_leafs

      if node.state == None:
          name = node.threshold_name
          threshold = node.threshold_value
          if_counts += 1
          print("{}if ({} <= {}){{".format(indent, name, threshold))
          recurse(node.left_node, depth + 1)
          print("{}}}".format(indent))
          print("{}else{{".format(indent, name, threshold) )
          recurse(node.right_node, depth + 1)
          print("{}}}".format(indent))
      else:
          total_leafs += 1
          print("{}return '{}';".format(indent, node.state))

  recurse(tree, 1)
  print("}")
  return if_counts, total_leafs

def prune_tree_to_code_in_python(tree, feature_names):
  """Prints a given DecisionTreeClassifier removing redundant conditions, in 
  Python3 code.

  Parameters
  ----------
  tree : DecisionTreeClassifier
      The model intended to convert into code
  feature_names : list[str]
      List containing the names of the features of the tree, in order

  Returns
  ----------
  (int, int)
      A tuple containing the number of decision nodes and the number of leaf 
      nodes, respectively

  """
  p_tree = _tree_to_tree(tree, feature_names)
  return _prunned_tree_to_code_in_python(p_tree, feature_names)

def prune_tree_to_code_in_c(tree, feature_names):
  """Prints a given DecisionTreeClassifier removing redundant conditions, in C 
  code.

  Parameters
  ----------
  tree : DecisionTreeClassifier
      The model intended to convert into code
  feature_names : list[str]
      List containing the names of the features of the tree, in order

  Returns
  ----------
  (int, int)
      A tuple containing the number of decision nodes and the number of leaf 
      nodes, respectively

  """
  p_tree = _tree_to_tree(tree, feature_names)
  return _prunned_tree_to_code_in_c(p_tree, feature_names)


def load_model(file_name):
  """Loads a object from a file persisted with joblib.dump

  Works under the same restrictions of joblib.load. Refer to 
  https://joblib.readthedocs.io/en/latest/generated/joblib.load.html

  Parameters
  ----------
  file_name : str, pathlib.Path, or file object
      The file or path from which to load the object

  Returns
  ----------
  Object
      The object stored in the file

  """
  clf = load(file_name)
  return clf

def tree_to_raw_code_in_python(tree, feature_names):
  """Prints a given DecisionTreeClassifier as it is, in Python3 code.

  Parameters
  ----------
  tree : DecisionTreeClassifier
      The model intended to convert into code
  feature_names : list[str]
      List containing the names of the features of the tree, in order

  Returns
  ----------
  (int, int)
      A tuple containing the number of decision nodes and the number of leaf 
      nodes, respectively

  """

  tree_ = tree.tree_
  feature_name = [
      "_".join(feature_names[i].split()) if i != _tree.TREE_UNDEFINED else "undefined!"
      for i in tree_.feature
  ]
  print("def tree({}):".format(", ".join(["_".join(i.split()) for i in feature_names])) )

  if_counts = 0
  total_leafs = 0

  def recurse(node, depth):
      indent = "  " * depth
      nonlocal if_counts
      nonlocal total_leafs

      if tree_.feature[node] != _tree.TREE_UNDEFINED:
          name = feature_name[node]
          threshold = tree_.threshold[node]
          if_counts += 1
          print("{}if {} <= {}:".format(indent, name, threshold))
          recurse(tree_.children_left[node], depth + 1)
          print("{}else:  # if {} > {}".format(indent, name, threshold) )
          recurse(tree_.children_right[node], depth + 1)
      else:
          total_leafs += 1
          print("{}return '{}'".format(indent, tree.classes_[np.argmax(tree_.value[node][0])]))


  recurse(0, 1)
  return if_counts, total_leafs

def tree_to_raw_code_in_c(tree, feature_names):
  """Prints a given DecisionTreeClassifier as it is, in C code.

  Parameters
  ----------
  tree : DecisionTreeClassifier
      The model intended to convert into code
  feature_names : list[str]
      List containing the names of the features of the tree, in order

  Returns
  ----------
  (int, int)
      A tuple containing the number of decision nodes and the number of leaf 
      nodes, respectively

  """

  tree_ = tree.tree_
  feature_name = [
      "_".join(feature_names[i].split()) if i != _tree.TREE_UNDEFINED else "undefined!"
      for i in tree_.feature
  ]
  print("char tree({}){{".format(", ".join(["_".join(i.split()) for i in feature_names])) )

  if_counts = 0
  total_leafs = 0

  def recurse(node, depth):
      indent = "  " * depth
      nonlocal if_counts
      nonlocal total_leafs

      if tree_.feature[node] != _tree.TREE_UNDEFINED:
          name = feature_name[node]
          threshold = tree_.threshold[node]
          if_counts += 1
          print("{}if ({} <= {}){{".format(indent, name, threshold))
          recurse(tree_.children_left[node], depth + 1)
          print("{}}}".format(indent))
          print("{}else{{".format(indent, name, threshold) )
          recurse(tree_.children_right[node], depth + 1)
          print("{}}}".format(indent))
      else:
          total_leafs += 1
          print("{}return '{}';".format(indent, tree.classes_[np.argmax(tree_.value[node][0])]))


  recurse(0, 1)
  print("}")
  return if_counts, total_leafs