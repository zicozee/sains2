# -*- coding: utf-8 -*-

# modules for importing and manipulating data
import numpy as np
import pandas as pd
from sainsbury_discontinued.config import config
from sklearn.utils import shuffle

missing_values = ["na", "missing", "n/a", "NA", "NAN", "nan", "NaN"]

# Training data
data1 = pd.read_csv(
    "research_phase/CatologueDiscontinuation.csv", na_values=missing_values)

# product info
data2 = pd.read_csv("research_phase/ProductDetails.csv",
                    na_values=missing_values)


# merge both csv files on customer_id
df = pd.merge(data1, data2, left_on='ProductKey',
              right_on='ProductKey', how='left')

# remove rows with missing target variable
df = df[~df['DiscontinuedTF'].isnull()]

# save data in dataset directory
file_name = 'training_data.csv'
df.to_csv(f"{config.DATASET_DIR}/{file_name}", encoding='utf-8')
