# config.py

import os
import pandas as pd
# Define base directory - this is at the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define other directories relative to the base directory
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

tissue_classification = {
  'MSC': 'Normal',
  'SKN': 'Normal',
  'FAT': 'Normal',
  'LEM': 'Benign',
  'MLS': 'Malignant',
  'PLS': 'Malignant',
  'LEI': 'Malignant',
  'HMS': 'Malignant',
}

metadata = pd.read_csv(os.path.join(METADATA_DIR, 'metadata.csv'))
