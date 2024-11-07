import pandas as pd
import json
import os

# Load the Knowledge Base
data_path = "../../datasets/FB15k-237"
dataset_path = os.path.join(data_path, "data_valid.txt")
entities_path = os.path.join(data_path, "data_FB15k_mid2name.txt")

triplets_df = pd.read_csv(dataset_path, sep="\t", header=None, names=["head", "relation", "tail"])
entities = pd.read_csv(entities_path, sep="\t", header=None, names=["mid", "entity"])

# Load the QA dataset
dataset_path = '../../datasets/FreebaseQA/FreebaseQA-eval.json'

with open(dataset_path, "r") as dataset:
    data = json.load(dataset)
# q_counter = 0
# parse_counter = []
# for q in data.get("Questions"):
#     parses = q.get("Parses")
    
    
qa_df = pd.DataFrame.from_dict(data.get("Questions"))
