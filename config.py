from decouple import config


DATA_DIR = config('DATA_DIR', default='/kaggle/input/proteinchallenge')
OUTPUT_DIR = config('OUTPUT_DIR', default='kaggle/working')