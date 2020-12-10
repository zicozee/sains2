# -*- coding: utf-8 -*-

from sainsbury_discontinued.processing import data_management as dm
from sainsbury_discontinued.config import config


def make_predicitons(input_data):
    _discontinued_identifier = dm.load_pipeline(file_name=config.PIPELINE_PATH)
    result = _discontinued_identifier.predict(input_data)
    prob = _discontinued_identifier.predict_proba(input_data)
    prob_1 = prob[:, 1]

    return result, prob_1


if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # load training data
    data = dm.load_dataset(file_name=config.TRAINING_DATA_FILE)

    # # train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.3, random_state=1984, stratify=data[config.TARGET]
    )

    # prediction and propensity
    pred, prob_1 = make_predicitons(data[config.FEATURES])
    data['pred'] = pred
    data['prob_1'] = prob_1

    # # determine classifcation report
    print(classification_report(y_test, pred))
