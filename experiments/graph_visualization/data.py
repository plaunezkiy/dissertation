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


qa_df = pd.DataFrame.from_dict(data.get("Questions"))

class QAItem:
    class QParse:
        class Answer:
            def __init__(self, obj):
                self.mid = obj["AnswersMid"]
                self.name = obj["AnswersName"]
        
        def __init__(self, obj):
            self.ID = obj["Parse-Id"]
            self.chain = obj["InferentialChain"]
            self.answers = [self.Answer(o) for o in obj["Answers"]]

    def __init__(self, obj):
        self.ID = obj["Question-ID"]
        self.question = obj["RawQuestion"]
        self.parses = [self.QParse(o) for o in obj["Parses"]]

# qa_items = []
for i, q in qa_df.iterrows():
    # qa_items.append()
    qa_item = QAItem(q)
    print(qa_item.parses)
    break


