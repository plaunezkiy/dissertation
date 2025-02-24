import pandas as pd
import csv
from tqdm import tqdm
from utils.llm.mistral import MistralLLM
from utils.file import export_results_to_file


mistral = MistralLLM()

for hop in ["1hop", "2hop", "3hop"]:
    print(hop)
    metaqa = pd.read_csv(f"/datasets/MetaQA/{hop}/qa_test.txt", sep="\t", header=None)
    metaqa.rename(columns={0: "Question", 1: "Answers"}, inplace=True)
    metaqa.Answers = metaqa.apply(lambda t: t.Answers.split("|"), axis=1)
    
    bline_path = f"/datasets/MetaQA/results/{hop}/bline.csv"
    bline = pd.read_csv(bline_path)
    l = len(bline)
    baseline_results = list(bline.Model.values)


    LEN_LIMITED_PROMPT = """
    You should answer the question below in under a sentence with no other text but the answer.
    If there are multiple answers, separate the answers by "|".
    Do not include "Answer:". Produce the answer only.
    Question:
    {question}
    """

    for i, r in tqdm(list(metaqa.iterrows())):
        if i < l:
            continue
        q = r.Question
        # print("Question:", q)
        # print("Answer:", a)
        prompt = LEN_LIMITED_PROMPT.format(question=q)
        # print("Prompt:", prompt)
        response = mistral.get_response(prompt)
        # print("Model:", response)
        baseline_results.append(response)
        
        if i % 250 == 0:
            export_results_to_file(bline_path, baseline_results)
    export_results_to_file(bline_path, baseline_results)