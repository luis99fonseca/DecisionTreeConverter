# DecisionTreeConverter

Module inspired by solution found in https://stackoverflow.com/a/39772170

Given a DecisionTreeClassifier model (from SKLearn - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), parses it to readable python3 or C code. In the case of C code, argument types must be manually written on the header.

Optionally, it is also possible to remove redundant conditions in the parse tree, as in (simplified):

```
def tree(arg01, arg02):
  if arg01 <= 18.5:
    if arg01 <= 1.5:
      return X
  ...
```

becomes

```
def tree(arg01, arg02):
  if arg01 <= 18.5:
      return X
  ...
```

This effectively reduces the number of leaf nodes and decision nodes, whenever possible.

## Example of Usage

```
clf = load_model('treeModel.joblib')
feats = ["arg01", "arg02"]
ifs, leafs = tree_to_raw_code_in_python(clf, feats)
```
