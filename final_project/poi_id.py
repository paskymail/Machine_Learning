#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("./Machine_Learning/tools/")
sys.path.append("./Machine_Learning/final_project/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

email_related = ["to_messages", "from_poi_to_this_person", "from_messages", "other", "from_this_person_to_poi", "shared_receipt_with_poi"]
finance_related = ["salary", "deferral_payments","total_payments", "loan_advances", "bonus", "restricted_stock_deferred", "deferred_income", "total_stock_value", "expenses", "exercised_stock_options","long_term_incentive", "restricted_stock", "director_fees"]
features_list = np.concatenate([["poi"], email_related, finance_related])


with open("./Machine_Learning/final_project/final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# The fist outlier is "TOTAL", which it is just the total values. We remove the value
data_dict.pop( "TOTAL", 0 )

# We are now going to see the outlier by email related and finance related.

# Some high salaries, bonus... are from persons of interest. Therefore we are going to do not cosider them as outliers.
# We remove them from out set first as their number is low

df = pd.DataFrame.from_dict(data_dict, orient = "index")
poi_df = df[df["poi"]== 1]

print(df)
print(poi_df)

my_df =df[df["poi"]!= 1]
my_df = my_df.replace("NaN", np.nan)
email_df = my_df[email_related]
finance_df = my_df[finance_related]

# We can see that we have outliers in the upper values. Let's take a look at them

NaN_table = pd.DataFrame()

for column in features_list:
    NaN_table[column] = [df[df[column]== "NaN"][column].count()]

print(NaN_table.T)


for column in finance_df:
    print(column)
    print(finance_df.nlargest(5, column)[column])
    print(finance_df.nsmallest(5, column)[column])

for column in email_df:
    print(column)
    print(email_df.nlargest(5, column)[column])
    print(email_df.nsmallest(5, column)[column])


pd.plotting.scatter_matrix(finance_df, figsize = (10, 10))
pd.plotting.scatter_matrix(email_df, figsize = (10, 10))

plt.show()

#Fianancial data outliers removal
data_dict["BHATNAGAR SANJAY"]["restricted_stock_deferred"] = np.NaN
data_dict["BHATNAGAR SANJAY"]["restricted_stock"] = np.NaN

data_dict["BELFER ROBERT"]["restricted_stock_deferred"] = np.NaN
data_dict["BELFER ROBERT"]["deferral_payments"] = np.NaN
data_dict["BELFER ROBERT"]["total_stock_value"] = np.NaN


# ### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict


#3.1 Choose features

### Extract features and labels from dataset for local testing


features_list = np.concatenate([["poi"], email_related, finance_related])

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# We are going to use kbest to select the features. We use f_classif as we are in a classfication problem. 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(features)
n_features = imp.transform(features)

selector = SelectKBest(f_classif, k="all")
selector.fit(X=n_features, y=labels)
print("features scores")
fo = np.concatenate([email_related, finance_related])
F_table = np.transpose([fo, selector.scores_])

print(F_table)

# The features with F-value higher than 5 and that will be kept are:

features_list = ["poi",'from_poi_to_this_person', "shared_receipt_with_poi",'salary','total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock']


# ### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# we change NaN for mean values per feature

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(features)
n_features = imp.transform(features)

# 3.3 We scale the features

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(n_features)

features = scaler.transform(n_features)

#3.4 Create a new feature

# As the financial figures are correlated between them, we will create a new feature that tries to summarize them.

financial_features_list = ['salary','total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock']

data = featureFormat(my_dataset, financial_features_list, sort_keys = True)
financial_labels, financial_features = targetFeatureSplit(data)

imp.fit(financial_features)
f_features = imp.transform(financial_features)
scaler.fit(f_features)
fin_features = scaler.transform(f_features)

New_fin_feature =[]

for row in fin_features:
    New_fin_feature.append(row.sum()/(row != 0).sum())

selector.fit(X=np.array(New_fin_feature).reshape(-1, 1), y=financial_labels)
print("New feature score")
print(selector.scores_)

# ### Task 4: Try a varity of classifiers
# ### Please name your classifier clf for easy export below.
# ### Note that if you want to do PCA or other multi-stage operations,
# ### you'll need to use Pipelines. For more info:
# ### http://scikit-learn.org/stable/modules/pipeline.html

# # Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, tree
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

def get_metrics(clf, features_train, features_test, labels_train, labels_test):
    clf.fit(features_train, labels_train)
    y_pred = clf.predict(features_test)
    precision = precision_score(labels_test, y_pred, zero_division= 0)
    recall = recall_score(labels_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(labels_test, y_pred)

    return precision, recall, balanced_accuracy

A_table = pd.DataFrame(columns=["precision", "recall", "balanced_accuracy"])


clf = GaussianNB()
A_table.at["GaussianNB"]= get_metrics(clf, features_train, features_test, labels_train, labels_test)

clf = tree.DecisionTreeClassifier()
A_table.at["DecisionTreeClassifier"]= get_metrics(clf, features_train, features_test, labels_train, labels_test)

clf= svm.LinearSVC()
A_table.at["SVC-Linear"]= get_metrics(clf, features_train, features_test, labels_train, labels_test)

clf = svm.SVC(kernel="rbf")
A_table.at["SVC-rbf"]= get_metrics(clf, features_train, features_test, labels_train, labels_test)


print(A_table)

# ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
# ### using our testing script. Check the tester.py script in the final project
# ### folder for details on the evaluation method, especially the test_classifier
# ### function. Because of the small size of the dataset, the script uses
# ### stratified shuffle split cross validation. For more info: 
# ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# # Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

def compoundmetric(estimator, features_test, labels_test ):
    y_pred = estimator.predict(features_test)
    precision = precision_score(labels_test, y_pred, zero_division= 0)
    recall = recall_score(labels_test, y_pred)

    return precision*recall


C_range = np.logspace(-3, 3, 7)
G_range = np.logspace(-3, 3, 7)
param_grid = {"gamma":G_range, 'C':C_range}
folds = 1000
cv = StratifiedShuffleSplit(n_splits=folds, test_size=0.3, random_state=42)
grid = GridSearchCV(svm.SVC("rbf"), scoring= compoundmetric, param_grid=param_grid, cv=cv)

grid.fit(features, labels)
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

#Due to the bad results, we keep the GaussianNB algorithm

clf = GaussianNB()

#Due to the fact that tester.py cannot cope with non values, we will include a SimpleImputer and our classifier in a pipeline
from sklearn.pipeline import Pipeline

estimators = [('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('clf', GaussianNB())]
pipe = Pipeline(estimators)

import tester

tester.test_classifier(pipe, my_dataset, features_list, folds = 1000)

clf.fit(features, labels)
y_pred =clf.predict(features)


# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# ### check your results. You do not need to change anything below, but make sure
# ### that the version of poi_id.py that you submit can be run on its own and
# ### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(pipe, my_dataset, features_list)
print("end")