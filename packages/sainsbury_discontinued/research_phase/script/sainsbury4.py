# -*- coding: utf-8 -*-

# modules for importing and manipulating data
import numpy as np
import pandas as pd
import re

# modules for modelling and scoring
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score

# modules for feature engineering and balancing training data
from imblearn.over_sampling import RandomOverSampler

# modules for visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns


# recover original matplotlib settings
mpl.rcParams.update(mpl.rcParamsDefault)

# to display all the columns of the dataframe
pd.pandas.set_option('display.max_columns', 100)

missing_values = ["na", "missing", "n/a", "NA", "NAN", "nan", "NaN"]
cat = pd.read_csv("research_phase/CatologueDiscontinuation.csv",
                  na_values=missing_values)
prod = pd.read_csv("research_phase/ProductDetails.csv",
                   na_values=missing_values)

# merge both csv files on product key
df = pd.merge(cat, prod, left_on='ProductKey',
              right_on='ProductKey', how='left')

# check class distribution
df['DiscontinuedTF'].value_counts()

# remove row if label is missing
df = df[~df['DiscontinuedTF'].isnull()]


# compare unique products with observations
print('Number of unique products: ', len(df['ProductKey'].unique()))
print('Number of observations in the Dataset: ', len(df))


# make a list of the variables that contain missing values
vars_with_na = [var for var in df.columns if df[var].isnull().sum() > 0]

# determine percentage of missing values: (seems to be no missing values)
df[vars_with_na].isnull().mean()


# create dict of {col_name: col unique values and counts}
dict_va = {}
for i in df.columns.tolist():
    dict_va.update({str(i): df[i].value_counts()})

# create a list of bool values
bools = [var for var in df.columns if df[var].dtypes == 'bool']

# convert bool to int
df[bools] = df[bools].astype(int)

# find product info i.e. supplier brand and product grouping
prod_info = [
    x for x in df.columns[df.columns.str.contains('Supplier|Hierarchy')]]

# catalogue id and product key
ID_vars = ['ProductKey', 'CatEdition']

# capture categorical variables in a list that are not productkey or cat id
cat_vars = [var for var in df.columns if df[var].dtypes ==
            'O' and var not in ID_vars]
print('Number of categorical variables: ', len(cat_vars))

# check unique value
df[cat_vars].nunique()

# check cardinality of these categorical variables
for col in cat_vars:
    print(str(col), ':', df[col].value_counts())


# eyeball distributions of numerical data
def cat_perclass(data, var):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['orange', 'black'])
    sns.catplot(x=var, hue='DiscontinuedTF', kind="count", data=data)
    plt.title(var, fontsize=14)
    plt.ylabel(var)
    plt.grid(b=None)
    plt.show()


plt.figure(figsize=(20, 6))

for col in cat_vars:
    cat_perclass(df, col)


# capture all numerical variables in alist
num_vars = [var for var in df.columns if df[var].dtypes !=
            'O' and var not in prod_info]

# capture all potential discrete variables in a list
discrete_vars = [var for var in num_vars if len(
    df[var].unique()) < 20 and var != 'ProductKey' and var != 'DiscontinuedTF']


# check cardinality of these discrete variables
df[discrete_vars].nunique()

for col in discrete_vars:
    print(str(col),
          df[col].value_counts())


# make list of continuous variables
cont_vars = [
    var for var in num_vars if var not in discrete_vars and var != 'DiscontinuedTF' and var not in (prod_info and ID_vars)]

# check statisics i.e. mean,max, min etc
df[cont_vars].describe()


# eyeball distributions of numerical data
def data_distro(data, var):
    data[var].hist()
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['orange'])
    plt.title(var, fontsize=14)
    plt.ylabel(var)
    plt.grid(b=None)
    plt.show()


plt.figure(figsize=(20, 6))
for var in cont_vars:
    data_distro(df, var)


#  find outliers within continous data.
# columns with outliers : actualsperweek, forcastperweek, salesprice,
def find_outliers(data, var):
    data = data.copy()
    sns.boxplot(x=data[var], palette="YlOrBr", linewidth=2.5, width=0.1)
    plt.title(var, fontsize=14)
    plt.show()


plt.figure(figsize=(20, 6))
for var in cont_vars:
    find_outliers(df, var)


# plot correlations between all the features in the data
# correlated features:[actualsperweek, forcastperweek]
def findCorrfeat(X):
    corr = X.corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr, cmap=cmap, square=True, cbar=False, cbar_kws={'shrink': 1},
        annot=True, annot_kws={'fontsize': 14}
    )
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)


plt.figure(figsize=(20, 10))
findCorrfeat(df)


# create training and testing data
X_train, X_test, y_train, y_test = train_test_split(df, df['DiscontinuedTF'],
                                                    test_size=0.3, stratify=df['DiscontinuedTF'],
                                                    random_state=1984, shuffle=True)


# save id's for later use if needed
train_prod, train_cat = X_train['ProductKey'], X_train['CatEdition']
test_prod, test_cat = X_test['ProductKey'], X_test['CatEdition']


X_train.drop(ID_vars, inplace=True, axis=1)
X_test.drop(ID_vars, inplace=True, axis=1)


# although there are no missing values, This is one way to deal with issues of missing values and potential inf values
# fill missing values in of continous columns with mean (or median as the some distributions are skewed) and remove any infinite values
# for feature in cont_vars:

#     X_train[feature].fillna(X_train[feature].mean(), inplace=True)
#     X_train[feature]=X_train[feature].replace([np.inf,-np.inf],0)
#     X_test[feature].fillna(X_train[feature].mean(), inplace=True)
#     X_test[feature]=X_test[feature].replace([np.inf,-np.inf],0)


# # fill discrete missing values with 0
# for feature in discrete_vars:
#     X_train[feature].fillna(0, inplace=True)
#     X_test[feature].fillna(0, inplace=True)


# althought categories are already relativley clean
# This is one way to deal with issues of missing values and standardisation of categorical features

# 1) fill missing values with unknown.
# 2) convert all letters to lower case.
# 3) remove redundant symbols
# def cleaning_cat_vars(data,var):
#     for i in var:
#         data[i].fillna("unknown",inplace=True)
#         data[i] = [str(x).lower() for x in data[i]]
#         data[i] = data[i].apply(lambda x: re.sub(r'\W', '',x))

#     return data

# X_train= cleaning_cat_vars(X_train,cat_vars)
# X_test= cleaning_cat_vars(X_test,cat_vars)


#  we need to reduce cardinality
# also we need to group rare variables to deal with the possibility of new label being added in the test
# or future live data that is not in training data

# There is no rule of thumb to determine how small is a small percentage to consider a label as rare,
# so we plot the data and make an eductated guess the optimal rare percentage

def plot_rare_threshold(df, cats):

    for col in cats:

        temp_df = pd.Series(df[col].value_counts() / len(df))

        # make plot with the above percentages
        fig = temp_df.sort_values(ascending=False).plot.bar()
        fig.set_xlabel(col)
        # orange for sainsbury theme
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['orange'])
        # add a line at 0.1% to flag the threshold for rare categories
        fig.axhline(y=0.001, color='blue')
        fig.set_ylabel('Percentage of products')
        plt.show()


cats2 = ['Status', 'Supplier', 'HierarchyLevel1', 'HierarchyLevel2', 'DIorDOM']
plot_rare_threshold(df, cats2)


# find and replace rare labels with rare and replace them with 'Rare'

def find_frequent_labels(data, var, rare_perc):

    # function finds the labels that are shared by more than
    # a certain % of the cars in the dataset

    data = data.copy()

    tmp = data.groupby(var)['DiscontinuedTF'].count() / len(data)

    return tmp[tmp > rare_perc].index


for var in cats2:

    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, 0.001)

    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(
        frequent_ls), X_train[var], 'Rare')

    X_test[var] = np.where(X_test[var].isin(
        frequent_ls), X_test[var], 'Rare')


# Ordering the categories according to the target means, i.e. assigning a number to the category from 1 to k
# where k is the number of distinct categories in the variable
# this numbering is informed by the mean of the target for each category.
# this creates a monotonic relationship between the variable and the target.


def categories_to_ordered(df_train, df_test, cats):

    # make a temporary copy of the datasets
    df_train_temp = df_train.copy()
    df_test_temp = df_test.copy()

    for col in cats:

        # order categories according to target mean
        ordered_labels = df_train_temp.groupby(
            [col])['DiscontinuedTF'].mean().sort_values().index

        # create the dictionary to map the ordered labels to an ordinal number
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

        # remap the categories  to these ordinal numbers
        df_train_temp[col] = df_train[col].map(ordinal_label)
        df_test_temp[col] = df_test[col].map(ordinal_label)

    # remove the target
    df_train_temp.drop(['DiscontinuedTF'], axis=1, inplace=True)
    df_test_temp.drop(['DiscontinuedTF'], axis=1, inplace=True)

    return df_train_temp, df_test_temp


X_train_ordered, X_test_ordered = categories_to_ordered(
    X_train, X_test, cats2)

X_train_ordered.head()


# dealing with correlated features
# #subtracting from each other i.e. forecasted - actuals

X_train_ordered['diffrence'] = X_train_ordered['ForecastPerWeek'] - \
    X_train_ordered['ActualsPerWeek']
X_test_ordered['diffrence'] = X_test_ordered['ForecastPerWeek'] - \
    X_test_ordered['ActualsPerWeek']

# drop features
X_train_ordered.drop(
    ['ForecastPerWeek', 'ActualsPerWeek'], inplace=True, axis=1)
X_test_ordered.drop(['ForecastPerWeek', 'ActualsPerWeek'],
                    inplace=True, axis=1)


# Unbalanced dataset: Strategy used here is oversampling while ensuring the majority class still remains the majority
sm = RandomOverSampler(sampling_strategy=0.8, random_state=1984)
X_train1, y_train1 = sm.fit_sample(X_train_ordered, np.ravel(y_train))

print("after over sampling, count of label '1' : {}".format(sum(y_train1 == 1)))
print("after over sampling, count of label '0' : {}".format(sum(y_train1 == 0)))


# 1) Base model: decision tree
# Use decsion tree as a base model from which you can gage how well more powerful algo's are performing
dt = DecisionTreeClassifier(
    random_state=1984, max_depth=8, min_samples_split=15)
dt.fit(X_train1, y_train1)

# Cross-validation is a resampling procedure used to evaluate machine learning models, typically limited data sample.
# The following splits data into 10 groups to train model on 9 and test on 1 at each round

scores = cross_val_score(dt, X_train1, y_train1, cv=10, scoring='f1_macro')
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                          scores.std()),
      end="\n\n")

prediction_dt1 = dt.predict(X_test_ordered)


# recursive feature elimination: wrapper approach (backward elimination)
selector_dt = RFECV(dt, cv=2)

# fit training data
selector_dt.fit(X_train1, y_train1)

# select best features
selected_dt = X_train1.columns[(selector_dt.get_support())]

# fit new reduced training data
dt.fit(X_train1[selected_dt], y_train1)

# make prediction and evalute model perfomance before and after feature selection
prediction_dt1 = dt.predict(X_test_ordered)
prediction_dt2 = dt.predict(X_test_ordered[selected_dt])

# precision recall f1
print(classification_report(y_test, prediction_dt1))
print(classification_report(y_test, prediction_dt2))

# accuracy
print(accuracy_score(y_test, prediction_dt1))
print(accuracy_score(y_test, prediction_dt2))


# 2) Ensemble : Bagging approach multiple trees trained in parellel on bootstrap sample of data and a subset of the features

# Random forest model with optimised parameters updated after grid search
model = RandomForestClassifier(
    n_estimators=800, random_state=1984, max_depth=8, min_samples_split=15, bootstrap=True)
# recursive feature elimination: wrapper approach (backward elimination)
selector_forest = RFECV(model, cv=2)

# # fit training data
# selector_forest.fit(X_train1, y_train1)

# # select best features
# selected_forest = X_train1.columns[(selector_forest.get_support())]

# # fit new reduced training data (on random forest specific features)
# model.fit(X_train1[selected_forest], y_train1)

# use selected dt to save time (on decsion tree specific features)
model.fit(X_train1[selected_dt], y_train1)

# make prediction and evalute model
# prediction_rf = model.predict(X_test_ordered[selected_forest])
prediction_rf = model.predict(X_test_ordered[selected_dt])
print(classification_report(y_test, prediction_rf))
print(accuracy_score(y_test, prediction_rf))


# # Exhaustive grid search: parameters for Random forest
# param_grid = {
#     'n_estimators': [400, 500,700,900,1200],
#     'max_features': ['auto', 'sqrt', 'log2',0.25, 0.5, 0.75, 1.0],
#     'max_depth' : [3,4,5,6,7,8],
#     'bootstrap': [False, True],
#     'criterion' :['gini', 'entropy'],
#     'min_samples_split': [2, 4, 6, 10, 20]
# }

# # key scores used: note accuracy is avoided due to the accuracy trap
# scoring = ['recall','f1','precision']

# # exhaustive grid search for random forest
# CV_rf = GridSearchCV(estimator=model, param_grid=param_grid, cv= 10, scoring = scoring, refit='f1')
# CV_rf.fit(X_train1[selected_features], y_train1)
# CV_rf.best_params_

# # use optimised model to make predictions and evaluate model
# prediction2=CV_rf.predict(X_test_ordered[selected_features])
# print(classification_report(y_test, prediction2))


# # 3)Ensemble : Boosting approach sequentially trained algorithm, each time placing more effort on the missclassified predictions

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=600, objective='binary:logistic',
#                     silent=True, nthread=1)

# selector_xgb = RFECV(xgb, step=1, cv=5)

# # fit training data
# selector_xgb.fit(X_train1, y_train1)

# # select best features
# # selected_xgb = X_train1.columns[(selector_xgb.get_support())]

# # fit new reduced training data
# # xgb.fit(X_train1[selected_xgb], y_train1)
# # use selected dt to save time
# xgb.fit(X_train1[selected_dt], y_train1)

# # make prediction and evalute model
# # = xgb.predict(X_test[selected_xgb])
# prediction_xgb = xgb.predict(X_test[selected_dt])
# print(classification_report(y_test, prediction_xgb))
# print(accuracy_score(y_test, prediction_xgb))


# # Exhaustive grid search: parameters for XGBoost
# params_grid_xgb = {
#         'n_estimators':[100,200,500,600], # how many trees to use
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5], # regularisation parameter this is added to the cover to help with pruning
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5],
#         'learning_rate' : [0.01,0.005,0.001] # (Discounts contribution of each tree in the sequence i.e. the learning rate helps us take a small step in the right direction
#         }


# # key scores used: note accuracy is avoided due to the accuracy trap
# scoring = ['recall','f1','precision']

# # exhaustive grid search
# CV_xgb= GridSearchCV(estimator=xgb, param_grid=params_grid_xgb, cv= 10, scoring = scoring, refit='f1')
# CV_xgb.fit(X_train1[selected_xgb], y_train1)
# CV_xgb.best_params_

# predicted_xgb = xgb.predict(X_test_ordered)
# print(classification_report(y_test, predicted_xgb))


# Plot confusion matrix
def Confusion_matrix(y, y_pred, classes, model_name):
    cnf_matrix1 = confusion_matrix(y, y_pred)
    c_train = pd.DataFrame(cnf_matrix1, index=classes, columns=classes)
    plt.subplot(2, 3, 3)
    ax = sns.heatmap(c_train, annot=True, cmap='Oranges', square=True, cbar=False,
                     fmt='.0f', annot_kws={"size": 9})
    plt.title('Confusion Matrix' + ' for ' + model_name, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)

    return(ax, cnf_matrix1)


classes = ['continued', 'Discontinued']
plt.figure(figsize=(20, 6))

# confusion matrix of rf and dt model
Confusion_matrix(y_test, prediction_rf, classes, 'Random Forest')
Confusion_matrix(y_test, prediction_dt, classes, 'Decision Trees')
plt.show()


# Create roc curve chart
# step1) compute predicted probablity
y_rf_proba = model.predict_proba(X_test_ordered[selected_forest])[::, 1]
y_dt_proba = dt.predict_proba(X_test_ordered[selected_dt])[::, 1]

# step2) compute roc and area under curve statstic
fpr_rf, tpr_rf, _ = roc_curve(y_test,  y_rf_proba)
fpr_dt, tpr_dt, _ = roc_curve(y_test,  y_dt_proba)

auc_rf = roc_auc_score(y_test, y_rf_proba)
auc_dt = roc_auc_score(y_test, y_dt_proba)

# step3) plot roc curve
# changing colours to match sainsbury's theme
plt.plot(fpr_rf, tpr_rf, color='orange',
         label="Random forest, auc="+str(np.round(auc_rf, 2)))
plt.plot(fpr_dt, tpr_dt, color='black',
         label="Decision Tree, auc="+str(np.round(auc_dt, 2)))

plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.legend(loc=4)
plt.title('Receiver Operating Characteristic Curve (ROC)', fontsize=13)
plt.xlabel('1 - specificity', fontsize=11)
plt.ylabel('sensitivity', fontsize=11)
plt.show()


# compute probabilty of a product belonging to each class
probability_rf = model.predict_proba(X_test_ordered[selected_forest])
probability_dt = dt.predict_proba(X_test_ordered[selected_dt])


# Plot the lift curve
# changing colours to match Sainsbury's theme
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["black", 'orange', 'grey'])

# decsion tree
skplt.metrics.plot_lift_curve(y_test, probability_dt)
plt.title('Decsion Tree', fontsize=13)
plt.grid(b=None)

# random forest
skplt.metrics.plot_lift_curve(y_test, probability_rf)
plt.title('Random Forest lift curve', fontsize=13)
plt.grid(b=None)
plt.show()
