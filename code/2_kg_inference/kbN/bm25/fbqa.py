import gc
import re
import csv
import os
import torch
import Stemmer
import pandas as pd
from tqdm import tqdm
from utils.graph import KGraphPreproc
from utils.graph.chain import GraphChain
from utils.llm.mistral import MistralLLM
from utils.prompt import GRAPH_QA_PROMPT, ENTITY_PROMPT


def get_fbqa_data(question_row):
    """
    Takes in a dataset row and returns Q and A as strings
    """
    question = question_row.Questions.get("RawQuestion", None)
    parse = question_row.Questions.get("Parses", [None])[0]
    if not parse:
        print(f"error in question: {question}")
        return question, None
    answer = parse.get("Answers")
    return question, answer


def get_response(prompt):
    global chain
    # del mistral
    gc.collect()
    torch.cuda.empty_cache()
    r = chain.invoke(prompt)
    return r["result"]


def save_results(fpath, data_rows):
    with open(fpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"])
        for r in data_rows:
            writer.writerow([str(r)])

####### load the qa, graph, llm
fbqa = pd.read_json("/datasets/FreebaseQA/FreebaseQA-eval.json")
fbkb_graph = KGraphPreproc.get_fbkb_graph()

mistral = MistralLLM()
chain = GraphChain.from_llm(
    llm=mistral,
    graph=fbkb_graph,
    qa_prompt=GRAPH_QA_PROMPT,
    entity_prompt=ENTITY_PROMPT,
    verbose=False,
)


for depth in [2,3]:
    # set the depth
    chain.exploration_depth = depth
    # init experiment
    experiment_name = f"kb{depth}"
    res_path = f"/datasets/FreebaseQA/results/{experiment_name}.csv"
    results = []
    l = 0
    # load if preinit'ed
    if os.path.isfile(res_path):
        r_df = pd.read_csv(res_path)
        l = len(r_df)
        results = list(r_df.Model.values)
    # load q's
    fbqa = pd.read_json("/datasets/FreebaseQA/FreebaseQA-eval.json")
    # run through
    for i, r in tqdm(list(fbqa.iterrows())):
        if i < l:
            continue
        q, a = get_fbqa_data(r)
        response = get_response(q)
        results.append(response)
        # backup every 10 qs
        if i % 10 == 0:
            save_results(res_path, results)
    save_results(res_path, results)

# python -m 2_kg_inference.kbN.fbqa &
# python -m 2_kg_inference.kbN.mqa &