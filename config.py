from decouple import config
import os


DATA_DIR = config('DATA_DIR', default='/kaggle/input/proteinchallenge')
OUTPUT_DIR = config('OUTPUT_DIR', default='/kaggle/working')
CACHE_DIR = config('CACHE_DIR', default='/kaggle/working/cache')
os.makedirs(CACHE_DIR, exist_ok=True)