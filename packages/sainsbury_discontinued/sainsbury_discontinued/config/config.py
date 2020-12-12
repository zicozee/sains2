import os
import pathlib


# create directories
# Package root
# PWD = os.path.dirname(os.path.abspath(__file__))
# PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))

# # dataset directory
# DATASET_DIR = os.path.join(PACKAGE_ROOT, 'datasets')

# # trained model directory
# TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')


PACKAGE_ROOT = pathlib.Path(sainsbury_discontinued.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"


# data
TRAINING_DATA_FILE = "training_data.csv"
TARGET = "DiscontinuedTF"


# variables included in analysis
FEATURES = ['DIorDOM',
            'SpringSummer',
            'WeeksOut',
            'Status',
            'SalePriceIncVAT',
            'ForecastPerWeek',
            'ActualsPerWeek',
            'Supplier',
            'HierarchyLevel1',
            'HierarchyLevel2',
            'Seasonal'
            ]

# boolean variables to impute and encode
BOOLS = ['SpringSummer', 'Seasonal']


# categorical variables to impute and encode
CATEGORICAL_VARS = ['DIorDOM', 'Status', 'Supplier',
                    'HierarchyLevel1', 'HierarchyLevel2']

# continous variables to impute with mean if needed
CONTINOUS_VARS = ['WeeksOut', 'SalePriceIncVAT',
                  'ForecastPerWeek', 'ActualsPerWeek']


# columns to be combined
FEATURES_TO_COMBINE1, FEATURES_TO_COMBINE2, COMBINED_FEATURE_NAME = 'ForecastPerWeek', 'ActualsPerWeek', 'diffrence'

# can be dropped after merger
DROP_FEATURES = ['ForecastPerWeek', 'ActualsPerWeek']


with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()


# pipeline name

PIPELINE_NAME = f'sainsbury_classfier_{_version}.pkl'
PIPELINE_PATH = os.path.join(TRAINED_MODEL_DIR, PIPELINE_NAME)
