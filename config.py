from decouple import config
from datatest import working_directory
import os
from pathlib import Path
import torch

with working_directory(__file__):
    current_dir = os.path.abspath('.')
    ROOT_DIR = Path(current_dir)
    KAGGLE = True if 'kaggle' in current_dir else False


# Specific config variables for Kaggle environment
if KAGGLE:
    DATA_DIR = '/kaggle/working/data'
    OUTPUT_DIR = '/kaggle/working'
    CACHE_DIR = '/kaggle/working/cache'
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'

# In a local environment, config variables can be overriden with a .env file at the root of the repository
else:
    DATA_DIR = config('DATA_DIR', default=ROOT_DIR/'data')
    OUTPUT_DIR = config('OUTPUT_DIR', default=ROOT_DIR/'output')
    CACHE_DIR = config('CACHE_DIR', default=ROOT_DIR/'cache')
    CHECKPOINT_DIR = config('CHECKPOINT_DIR', default=ROOT_DIR/'checkpoints')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
