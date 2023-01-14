from decouple import config
from datatest import working_directory
import os
from pathlib import Path

with working_directory(__file__):
    current_dir = os.path.abspath('.')
    ROOT_DIR = Path(current_dir)
    KAGGLE = True if 'kaggle' in current_dir else False


# Specific config variables for Kaggle environment
if KAGGLE:
    DATA_DIR = '/kaggle/input/proteinchallenge'
    OUTPUT_DIR = '/kaggle/working'
    CACHE_DIR = ROOT_DIR/'cache'

# In a local environment, config variables can be overriden with a .env file at the root of the repository
else:
    DATA_DIR = config('DATA_DIR', default=ROOT_DIR/'data')
    OUTPUT_DIR = config('OUTPUT_DIR', default=ROOT_DIR/'output')
    CACHE_DIR = config('CACHE_DIR', default=ROOT_DIR/'cacha')

os.makedirs(CACHE_DIR, exist_ok=True)