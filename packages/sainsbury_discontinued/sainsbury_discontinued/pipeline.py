# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sainsbury_discontinued.processing import preprocessors as pp
from sainsbury_discontinued.config import config
import logging


_logger = logging.getLogger(__name__)

# Oversampling to balance dataset
sm = RandomOverSampler(sampling_strategy=0.8,
                       random_state=1984)


# using imblearn pipeline : this allows us to use oversampling in training phase and skip it during predict phase
discontinued_pipe = Pipeline(
    [
        (
            "bool_Converter",
            pp.BoolConverter(variables=config.BOOLS),
        ),

        (
            "categorical_imputer",
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS),
        ),


        (
            "numerical_inputer",
            pp.NumericalImputer(variables=config.CONTINOUS_VARS),
        ),


        (
            "col_merger",
            pp.ColumnMerger(first=config.FEATURES_TO_COMBINE1,
                            second=config.FEATURES_TO_COMBINE2,
                            name=config.COMBINED_FEATURE_NAME),
        ),

        (
            "rare_Enc",
            pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS),
        ),

        (
            "categorical_enc",
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS),
        ),
        (
            "drop_features",
            pp.DropFeatures(variables_to_drop=config.DROP_FEATURES),
        ),
        (
            "over_sampling",
            sm,
        ),
        ("discontinued_classfier", RandomForestClassifier(n_estimators=900,
                                                          random_state=1984, max_depth=8,
                                                          min_samples_split=15, bootstrap=True)),
    ]
)
