# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import re
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin


# covert bool to 0 and 1's
class BoolConverter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].astype(int)
        return X


# categorical missing value imputer. Strategy: 'Unknown"
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna("unknown", inplace=True)
            X[feature] = [str(x).lower() for x in X[feature]]
            X[feature] = X[feature].apply(lambda x: re.sub(r'\W', '', x))
        return X


# Numerical missing value imputer. Strategy: mean
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # save mean in a dictionary
        self.imputer_dict_ = {}

        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mean()
        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
            X[feature] = X[feature].replace([np.inf, -np.inf], 0)
        return X


# rare label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.001, variables=None):

        self.tol = tol

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[
                feature]), X[feature], 'Rare')

        return X


# string to numbers categorical encoder(creates monotonic relationship)
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['Target']

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['Target'].mean(
            ).sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X


# highly correlated columns merger
class ColumnMerger(BaseEstimator, TransformerMixin):
    def __init__(self, first, second, name):
        self.first = first
        self.second = second
        self.name = name

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X[self.name] = X[self.first] - X[self.second]
        return X


# drop features deemed redundant by feature selection algorthims
class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X
