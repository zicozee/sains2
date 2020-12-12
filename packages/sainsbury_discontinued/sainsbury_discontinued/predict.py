# -*- coding: utf-8 -*-
import pandas as pd
import logging
from sainsbury_discontinued.processing import data_management as dm
from sainsbury_discontinued.config import config
from sainsbury_discontinued import __version__ as _version


_logger = logging.getLogger(__name__)


_discontinued_pipe = dm.load_pipeline(file_name=config.PIPELINE_PATH)


def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved random forest pipeline.

    Args:
        input_data: model prediction inputs in dict format

    Returns:
        1)Predictions for each input row, 
        2)Propensity for each input row 
        3)Model version.
    """
    # data = pd.read_json(input_data)
    data = pd.DataFrame(input_data)
    output = _discontinued_pipe.predict(data[config.FEATURES])
    prob = _discontinued_pipe.predict_proba(data[config.FEATURES])
    prob_1 = prob[:, 1]

    results = {"predictions": output,
               "propensity": prob_1, "version": _version}

    _logger.info(
        f"model version: {_version} "
        f"Inputs used: {data} "
        f"Predictions: {results}"
    )

    return results
