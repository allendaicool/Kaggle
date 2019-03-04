import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import numpy as np


def replaceNaNwithAvg(dataFrame, column_name):
	age_array = dataFrame[dataFrame[column_name]!=np.nan][column_name]
	dataFrame[column_name] = dataFrame[column_name].replace(np.nan,age_array.mean())
	return dataFrame

file_path_train =  'titanic/train.csv'
file_path_test = 'titanic/test.csv'

dataFrame = pandas.read_csv(file_path_train, usecols=['Survived', 'Pclass', 'Sex', 'Age', \
 	'SibSp', 'Parch', 'Fare', 'Embarked'])
dataFrame = dataFrame.replace({'female': 0, 'male': 1})
dataFrame = dataFrame.replace({'Embarked': {'C': 0, 'Q': 1, 'S':2}})
dataFrame = replaceNaNwithAvg(dataFrame, "Age")
dataFrame['Embarked'] = dataFrame['Embarked'].replace(np.nan,3)

dataArray = dataFrame.values

idx_output_columns=[0]
idx_feature_columns = [i for i in range(np.shape(dataArray)[1]) if i not in idx_output_columns]
feature_columns = dataArray[:,idx_feature_columns]
output_columns =  dataArray[:,0]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
print (np.shape(output_columns))
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
model.fit(feature_columns, output_columns)
# print ('accuracy is ',model.score(feature_columns, output_columns))
results = model_selection.cross_val_score(model, feature_columns, output_columns, cv=kfold)
print (results)
print('decision tree classifer is ' , results.mean())

num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, feature_columns, output_columns, cv=kfold)
model.fit(feature_columns, output_columns)
print('AdaBoostClassifier is ' , results.mean())




