import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.kernel_ridge import KernelRidge
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import string
import math
import sys

import numpy as np


#https://www.kaggle.com/volhaleusha/titanic-tutorial-encoding-feature-eng-81-8
# Create function that take name and separates it into title, family name and deletes all puntuation from name column:
def name_sep(data):
	families=[]
	titles = []
	new_name = []
	#for each row in dataset:
	for i in range(len(data)):
		name = data.iloc[i]
		# extract name inside brakets into name_bracket:
		if '(' in name:
			name_no_bracket = name.split('(')[0] 
		else:
			name_no_bracket = name
			
		family = name_no_bracket.split(",")[0]
		title = name_no_bracket.split(",")[1].strip().split(" ")[0]
		
		#remove punctuations accept brackets:
		for c in string.punctuation:
			name = name.replace(c,"").strip()
			family = family.replace(c,"").strip()
			title = title.replace(c,"").strip()
			
		families.append(family)
		titles.append(title)
		new_name.append(name)
			
	return families, titles, new_name 

def ticket_sep(data_ticket):
	ticket_type = []

	for i in range(len(data_ticket)):

			ticket =data_ticket.iloc[i]

			for c in string.punctuation:
				ticket = ticket.replace(c,"")
				splited_ticket = ticket.split(" ")   
			if len(splited_ticket) == 1:
				ticket_type.append('NO')
			else: 
				ticket_type.append(splited_ticket[0])
	return ticket_type 


def cabin_sep(data_cabin):
	cabin_type = []

	for i in range(len(data_cabin)):

			if data_cabin.isnull()[i] == True: 
				cabin_type.append('NaN') 
			else:    
				cabin = data_cabin[i]
				cabin_type.append(cabin[:1]) 
			
	return cabin_type

def decision_tree_adaboost_classifier(dataFrame, output_column):
	dataArray = dataFrame.values
	idx_output_columns=[output_column]
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
# decision tree classifer is  0.8316853932584269
# AdaBoostClassifier is  0.8417977528089887

def data_preprocess(data_train, data_test):
	data_train['Embarked'] = data_train['Embarked'].fillna("S")
	data_test['Fare'] = data_test['Fare'].fillna(data_train['Fare'].median())	

	# 1. create feature to show rows with missing values of age:
	data_train['Age_NA'] =np.where(data_train.Age.isnull(), 1, 0)
	data_test['Age_NA'] =np.where(data_test.Age.isnull(), 1, 0)	

		# create feature where missing age is imputed with mean of age values that are not missing
	data_train['Age_mean'] =np.where(data_train.Age.isnull(), data_train['Age'].mean(), data_train['Age'])
	data_test['Age_mean'] =np.where(data_test.Age.isnull(), data_test['Age'].mean(), data_test['Age'])

	data_train = data_train.drop(['PassengerId'], axis=1)
	data_test = data_test.drop(['PassengerId'], axis=1)	

	data_train["ticket_type"] = ticket_sep(data_train.Ticket)
	data_test["ticket_type"]= ticket_sep(data_test.Ticket)
	# for those types that have less than 15 samples in training set, assign type to 'OTHER':
	for t in data_train['ticket_type'].unique():
		if len(data_train[data_train['ticket_type']==t]) < 15:
			data_train.loc[data_train.ticket_type ==t, 'ticket_type'] = 'OTHER_T'

	for t in data_test['ticket_type'].unique():
		if t not in data_train['ticket_type'].unique():
			data_test.loc[data_test.ticket_type ==t, 'ticket_type']= 'OTHER_T'

		# where ticket_type is 'SOTONOQ' convert it to 'A5'
	data_train["ticket_type"] = np.where(data_train["ticket_type"]=='SOTONOQ', 'A5', data_train["ticket_type"])
	data_test["ticket_type"] = np.where(data_test["ticket_type"]=='SOTONOQ', 'A5', data_test["ticket_type"])
	# drop Ticket from dataset:

	data_train = data_train.drop(['Ticket'], axis=1)
	data_test = data_test.drop(['Ticket'], axis=1)

	# apply cabin sep on test and train set:
	data_train['cabin_type'] = cabin_sep(data_train.Cabin)
	data_test['cabin_type'] = cabin_sep(data_test.Cabin)
	# for those types that have less than 15 samples in training set, assign type to 'OTHER_C':

	for t in data_train['cabin_type'].unique():
		if len(data_train[data_train['cabin_type']==t]) <= 15:
			data_train.loc[data_train.cabin_type ==t, 'cabin_type'] = 'OTHER_C'
		   
		
	for t in data_test['cabin_type'].unique():
		if t not in data_train['cabin_type'].unique():
			data_test.loc[data_test.cabin_type ==t, 'cabin_type'] = 'OTHER_C'

	data_train = data_train.drop(['Cabin'], axis=1)
	data_test = data_test.drop(['Cabin'], axis=1)

		# apply name_sep on train and test set:
	data_train['family'], data_train['title'], data_train['Name']  = name_sep(data_train.Name)
	data_test['family'], data_test['title'], data_test['Name'] = name_sep(data_test.Name)

		# for those types that have less than 15 samples in training set, assign type to 'OTHER':	

	for t in data_train['title'].unique():
		if len(data_train[data_train['title']==t]) <= 15:
			data_train.loc[data_train.title ==t, 'title'] = 'OTHER'
		   
		
	for t in data_test['title'].unique():
		if t not in data_train['title'].unique():
			data_test.loc[data_test.title ==t, 'title'] = 'OTHER'

	#create a list with all overlapping families
	overlap = [x for x in data_train.family.unique() if x in data_test.family.unique()]

	# introduce new column to data called family_size:
	data_train['family_size'] = data_train.SibSp + data_train.Parch +1
	data_test['family_size'] = data_test.SibSp + data_test.Parch +1

	# calculate survival rate for each family in train_set:
	rate_family = data_train.groupby('family')['Survived', 'family','family_size'].median()

	# if family size is more than 1 and family name is in overlap list 
	overlap_family ={}
	for i in range(len(rate_family)):
		if rate_family.index[i] in overlap and  rate_family.iloc[i,1] > 1:
			overlap_family[rate_family.index[i]] = rate_family.iloc[i,0]

	mean_survival_rate = np.mean(data_train.Survived)
	family_survival_rate = []
	family_survival_rate_NA = []	

	for i in range(len(data_train)):
		if data_train.family[i] in overlap_family:
			family_survival_rate.append(overlap_family[data_train.family[i]])
			family_survival_rate_NA.append(1)
		else:
			family_survival_rate.append(mean_survival_rate)
			family_survival_rate_NA.append(0)
			
	data_train['family_survival_rate']= family_survival_rate
	data_train['family_survival_rate_NA']= family_survival_rate_NA

	mean_survival_rate = np.mean(data_train.Survived)
	family_survival_rate = []
	family_survival_rate_NA = []	

	for i in range(len(data_test)):
		if data_test.family[i] in overlap_family:
			family_survival_rate.append(overlap_family[data_test.family[i]])
			family_survival_rate_NA.append(1)
		else:
			family_survival_rate.append(mean_survival_rate)
			family_survival_rate_NA.append(0)
	data_test['family_survival_rate']= family_survival_rate
	data_test['family_survival_rate_NA']= family_survival_rate_NA

	# drop name and family from dataset:
	data_train = data_train.drop(['Name', 'family'], axis=1)
	data_test = data_test.drop(['Name', 'family'], axis=1)

		# calculate upper bound for Fair
	IQR = data_train.Fare.quantile(0.75) - data_train.Fare.quantile(0.25)
	upper_bound = data_train.Fare.quantile(0.75) + 3*IQR
	# for train and test sets convert all values in column Fair where age is more than upper_bound to upper_bound:
	data_train.loc[data_train.Fare >upper_bound, 'Fare'] = upper_bound 
	data_test.loc[data_test.Fare >upper_bound, 'Fare'] = upper_bound

		# calculate upper bound for Age_mean
	IQR = data_train.Age_mean.quantile(0.75) - data_train.Age_mean.quantile(0.25)
	upper_bound = data_train.Age_mean.quantile(0.75) + 3*IQR
	# for train and test sets convert all values in column Fair where age is more than upper_bound to upper_bound:
	data_train.loc[data_train.Age_mean >upper_bound, 'Age_mean'] = upper_bound 
	data_test.loc[data_test.Age_mean >upper_bound, 'Age_mean'] = upper_bound


	data = pd.concat([data_train.drop(['Survived'], axis=1), data_test], axis =0, sort = False)
	# encode variables into numeric labels
	le = LabelEncoder()	

	columns = ['Sex', 'Embarked', 'ticket_type', 'cabin_type', 'title']	

	for col in columns:
		le.fit(data[col])
		data[col] = le.transform(data[col])

		# drop columns that have information about age or are strongly correlated with other features
	data = data.drop(['Age_mean', 'Age_NA'], axis =1)

	x_train_age = data.dropna().drop(['Age'], axis =1)
	y_train_age = data.dropna()['Age']
	x_test_age = data[pd.isnull(data.Age)].drop(['Age'], axis =1)
	model_lin = make_pipeline(StandardScaler(),KernelRidge())
	kfold = model_selection.KFold(n_splits=10, random_state=4, shuffle = True)
		#model_lin.get_params().keys()
	parameters = {'kernelridge__gamma' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
				  'kernelridge__kernel': ['rbf', 'linear'],
				   'kernelridge__alpha' :[0.001, 0.01, 0.1, 1, 10, 100, 1000],
				  
				 }
	search_lin = GridSearchCV(model_lin, parameters, n_jobs = -1, cv = kfold, scoring = 'r2',verbose=1)
	search_lin.fit(x_train_age, y_train_age)

	y_test_age = search_lin.predict(x_test_age)
	data.loc[data['Age'].isnull(), 'Age'] = y_test_age
	idx = int(data_train.shape[0])
	data_train['Age'] = data.iloc[:idx].Age
	data_test['Age'] = data.iloc[idx:].Age

		# encode 'cabin_type' into numeric labels
	le = LabelEncoder()
	data_train_LE = data_train.copy()
	data_test_LE = data_test.copy()	

	columns = ['Sex', 'Embarked', 'ticket_type', 'cabin_type', 'title']	

	for col in columns:
		le.fit(data_train_LE[col])
		data_train_LE[col] = le.fit_transform(data_train_LE[col])
		data_test_LE[col] = le.transform(data_test_LE[col])
		

	drop_col = ['Age_mean', 'SibSp', 'Parch']
	data_train_LE = data_train_LE.drop(drop_col, axis=1)
	data_test_LE = data_test_LE.drop(drop_col, axis=1)

	decision_tree_adaboost_classifier(data_train_LE, 0)


if __name__ == "__main__":
	file_path_train =  'titanic/train.csv'
	file_path_test = 'titanic/test.csv'	
	data_train = pd.read_csv(file_path_train)
	data_test = pd.read_csv(file_path_test)	
	data_preprocess(data_train, data_test)












