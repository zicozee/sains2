# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
from sainsbury_discontinued.config import config 
from sainsbury_discontinued import pipeline
from sainsbury_discontinued.processing import data_management as dm
from sainsbury_discontinued import __version__ as _version
import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # load training data
    data = dm.load_dataset(file_name=config.TRAINING_DATA_FILE)

    # train and test split 
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.3, random_state=1984, stratify=data[config.TARGET]
    ) 
    
    # fit pipeline
    pipeline.discontinued_pipe.fit(X_train[config.FEATURES], y_train)
   
    # add model version to logs 
    _logger.info(f"saving model version: {_version}")
    
    # save pipeline
    dm.save_pipeline(pipeline_to_persist=pipeline.discontinued_pipe)


if __name__ == "__main__":
    run_training()
 