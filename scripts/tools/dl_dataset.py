import sys
from datasets import load_dataset

data_name = sys.argv[1]
out_dir = sys.argv[2]
data = load_dataset(data_name)
data.save_to_disk(out_dir)