import pandas as pd
import json
import os

# Load the Knowledge Base
data_path = "../../datasets/FB15k-237"
dataset_path = os.path.join(data_path, "data_valid.txt")
entities_path = os.path.join(data_path, "data_FB15k_mid2name.txt")

triplets_df = pd.read_csv(dataset_path, sep="\t", header=None, names=["head", "relation", "tail"])
entities = pd.read_csv(entities_path, sep="\t", header=None, names=["mid", "entity"])

def process_mid(mid):
    """
    provides consistency between id naming in the FB15k and FBQA
    """
    # replaces the / with . and drops the first slash
    return '.'.join(mid[1:].split("/"))

# inverse mapping {entity:mid}
entities_dict = {entities[entities.mid == mid].iloc[0].entity : process_mid(mid)  for mid in entities.mid}

# Load the QA dataset
dataset_path = '../../datasets/FreebaseQA/FreebaseQA-eval.json'

with open(dataset_path, "r") as dataset:
    data = json.load(dataset)


qa_df = pd.DataFrame.from_dict(data.get("Questions"))

found_entities = []
for i, q in qa_df.iterrows():
    tokens = q.RawQuestion.split()
    for token in tokens:
        if token in entities_dict:
            found_entities.append(token)
            print(i, q)
            print(token, entities_dict[token])
            print()
print(len(found_entities))

# inferential chain constructor
# 
# chain_df = pd.DataFrame(columns=["ID", "Chain", "ChainLength"])
# for i, q in qa_df.iterrows():
#     # qa_items.append()
#     # qa_item = QAItem(q)
#     # print(qa_item.parses)
#     for p in q["Parses"]:
#         chain_df.loc[i] = {"ID": q["Question-ID"], "Chain": p["InferentialChain"], "ChainLength": len(p["InferentialChain"].split("."))}
