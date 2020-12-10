# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.pipeline import Pipeline
from sainsbury_discontinued.config import config
from pathlib import Path
import joblib
import logging
import typing as t


_logger = logging.getLogger(__name__)

# load training dataset 
def load_dataset(*, file_name: str) -> pd.DataFrame:
    missing_values=["na","missing","n/a","NA","NAN","nan","NaN"]
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}", na_values=missing_values)
    _data[config.TARGET] = _data[config.TARGET].astype(int)
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
    """Saves the versioned model, and overwrites any previous
    saved models"""
    
    remove_old_pipelines(files_to_keep=[config.PIPELINE_NAME])
    joblib.dump(pipeline_to_persist,config.PIPELINE_PATH)
    _logger.info(f"saved pipeline: {config.PIPELINE_NAME}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    trained_model = joblib.load(filename=config.PIPELINE_PATH)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines o ensure there is a simple one-to-one
    mapping between the package version and the model, though we include 
    the previous version for diffrential testing"""

    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in Path(config.TRAINED_MODEL_DIR).iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

