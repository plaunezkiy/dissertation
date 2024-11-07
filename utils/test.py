import pandas as pd
import json
from datasets import load_dataset

ds = load_dataset("KGraph/FB15k-237")

dataset_path = 'datasets/FreebaseQA/FreebaseQA-eval.json'

with open(dataset_path, "r") as dataset:

    data = json.load(dataset)
q_counter = 0
parse_counter = []
for q in data.get("Questions"):
    parses = q.get("Parses")
    
    
df = pd.DataFrame.from_dict(data.get("Questions"))

# df = pd.read_json(dataset_path)
# print(df.iloc[0])