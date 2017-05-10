#IMPORT LIBRARIES TO BE USED
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import Binarizer, LabelBinarizer, FunctionTransformer, RobustScaler, Imputer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import f_regression, SelectFromModel

#READ IN THE TRAINING DATA AND SET Y ASIDE
df = pd.read_csv('university_train.csv')
y = df['percent_on_student_loan']
df = df.drop('percent_on_student_loan', axis = 1)

#LABELABLE COLUMNS
def common_deg(data):
	common = data['PREDDEG']
	return common.values.reshape(-1,1)

def max_deg(data):
	max_deg = data['HIGHDEG']
	return max_deg.values.reshape(-1,1)

def pub_priv(data):
	pub_priv = data['CONTROL']
	return pub_priv.values.reshape(-1,1)

def locale(data):
	locale = data['LOCALE']
	return locale.values.reshape(-1,1)

def carnegie_ug(data):
	carnegie_ug = data['CCUGPROF']
	return carnegie_ug.values.reshape(-1,1)

def carnegie_setting(data):
	carnegie_setting = data['CCSIZSET']
	return carnegie_setting.values.reshape(-1,1)

def hbcu(data):
	hbcu = data['HBCU']
	return hbcu.values.reshape(-1,1)

def pbi(data):
	pbi = data['PBI']
	return pbi.values.reshape(-1,1)

def mystery(data):
	mystery = data['RELAFFIL']
	return mystery.values.reshape(-1,1)

#CONTINUOUS COLUMNS
def undergrads(data):
	undergrads = data['UGDS']
	return undergrads.values.reshape(-1,1)

def female(data):
	female = data['FEMALE']
	return female.values.reshape(-1,1)

def age(data):
	age = data['AGE_ENTRY']
	return age.values.reshape(-1,1)

def married(data):
	married = data['MARRIED']
	return married.values.reshape(-1,1)
########PIPELINE ASSEMBLY
#LABELED FEATURES
pipe1 = make_pipeline(
	FunctionTransformer(common_deg, validate = False),
	LabelBinarizer())
pipe2 = make_pipeline(
	FunctionTransformer(max_deg, validate = False),
	LabelBinarizer())
pipe3 = make_pipeline(
	FunctionTransformer(pub_priv, validate = False),
	LabelBinarizer())
pipe4 = make_pipeline(
	FunctionTransformer(locale, validate = False),
	LabelBinarizer())
pipe5 = make_pipeline(
	FunctionTransformer(pbi, validate = False), 
	Binarizer())
pipe6 = make_pipeline(
	FunctionTransformer(carnegie_ug, validate = False),
	LabelBinarizer())
pipe_carnegie = make_pipeline(
	FunctionTransformer(carnegie_setting, validate = False),
	LabelBinarizer())
pipe_hbcu = make_pipeline(
	FunctionTransformer(hbcu, validate = False),
	Binarizer())
pipe_mystery = make_pipeline(
	FunctionTransformer(mystery, validate = False),
	LabelBinarizer())

#CONTINUOUS COLUMN VARIABLES
pipe7 = make_pipeline(
	FunctionTransformer(undergrads, validate = False))
pipe8 = make_pipeline(
	FunctionTransformer(female, validate = False))
pipe9 = make_pipeline(
	FunctionTransformer(age, validate = False))
pipe10 = make_pipeline(
	FunctionTransformer(married, validate = False))

union = make_union(
	pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, pipe_carnegie, pipe_hbcu, pipe_mystery, pipe7, pipe8, pipe9, pipe10)

fu = union.fit_transform(df)

x_train, x_test, y_train, y_test = train_test_split(fu, y, train_size = 0.75, random_state = 17)

def decisiontree_optimizer(x, y):
	dtr = DecisionTreeRegressor()
	br = BaggingRegressor(dtr, n_jobs = -1)
	grid = GridSearchCV(br,
		param_grid = {
		#The number of DecisionTreeRegressors to experiment with in each iteration
		'n_estimators': [100, 150, 200], 
		#The fraction of sample to use as training data for each DecisionTreeRegressor
		'max_samples': [.66, .75, .8, .9, 1.0],
		#The fraction of features to include in each training set
		'max_features': [.66, .75, .8, .9, 1.0],
		#Allow trees to reuse previous trees' solutions in the bag or do not allow it
		'warm_start': [True, False]},
		#The number of StratifiedKFolds to make 
		cv = 3, 
		#I prefer to have the computer print what it is doing
		verbose = 5)
	grid.fit(x, y)
	print("Highest R^2 score on training data: ", grid.best_score_)
	print("The parameters that yield the greatest score: ", grid.best_params_)

	return grid.best_params_

b_params = decisiontree_optimizer(x_train, y_train) 

dtr = DecisionTreeRegressor()
br = BaggingRegressor(
	dtr, 
	n_estimators = b_params['n_estimators'], 
	max_features = b_params['max_features'], 
	max_samples = b_params['max_samples'],
	warm_start = b_params['warm_start'])
br.fit(x_train, y_train)
print('R^2 on held out test set', br.score(x_test, y_test))

#The R^2 score maintained a significant level of consistency so now I would like to fit our entire dataset and make our submission to Kaggle

br.fit(fu, y)

test_df = pd.read_csv('university_test.csv')
fu_test = union.transform(test_df)
test_df['Prediction'] = br.predict(fu_test)
pd.DataFrame(test_df[["id_number", 'Prediction']]).to_csv('regression_submission.csv', index = False)











