import gc
import os
import torch
import gc
import torch
import pandas as pd
from tqdm import tqdm
import networkx as nx
import ast
from utils.graph import KGraphPreproc
from utils.graph.chain import GraphChain
from utils.llm.mistral import MistralLLM
from utils.prompt import GRAPH_QA_PROMPT, ENTITY_PROMPT
from utils.file import export_results_to_file

device = torch.device("cuda:1")
torch.cuda.set_device(device)
torch.set_default_device(device)

def get_fbqa_data(question_row):
    """
    Takes in a dataset row and returns Q and A as strings
    """
    question = question_row.get("RawQuestion", None)
    return question, None


def get_response(prompt):
    global chain
    # del mistral
    gc.collect()
    torch.cuda.empty_cache()
    r = chain.invoke(prompt)
    return r["result"]

####### load the qa, graph, llm
fbkb_graph = KGraphPreproc.get_fbkb_graph()

mistral = MistralLLM()
chain = GraphChain.from_llm(
    llm=mistral,
    graph=fbkb_graph,
    qa_prompt=GRAPH_QA_PROMPT,
    entity_prompt=ENTITY_PROMPT,
    verbose=False,
)

def entity_path_len(entities):
    for path in entities:
        start = path[0]
        for target in path[1:]:
            try:
                return len(nx.shortest_path(
                    fbkb_graph._graph, start, target
                ))
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                continue
    return -1

fbqa = pd.read_csv("/datasets/FreebaseQA/FbQA-eval-1000.csv", index_col=0)
fbqa["entities"] = fbqa["entities"].apply(ast.literal_eval)
fbqa["hops"] = fbqa.apply(lambda t: entity_path_len(t["entities"]), axis=1)

delta = 1
for depth in [4, 5, 6, 7, 8, 9, 10, 11]:
    print(depth)
    # set the depth
    chain.exploration_depth = depth
    chain.ranking_strategy = "sbert"
    # init experiment
    experiment_name = f"sbert-kb{depth}"
    res_path = f"/datasets/FreebaseQA/results/{experiment_name}.csv"
    results = []
    id_list = []
    l = 0
    # load if preinit'ed
    if os.path.isfile(res_path):
        r_df = pd.read_csv(res_path)
        l = len(r_df)
        results = list(r_df.Model.values)
    # run through
    for c, (i, r) in enumerate(tqdm(list(fbqa.iterrows()))):
        id_list.append(i)
        if c < l:
            continue
        # define range (if outside, continue)
        if not (depth - delta <= r["hops"] <= depth + delta):
            results.append("")
            continue
        q, a = get_fbqa_data(r)
        response = get_response(q)
        results.append(response)
        # backup every 10 qs
        if c % 10 == 0:
            export_results_to_file(res_path, results, id_list)
    export_results_to_file(res_path, results, id_list)

# python -m 2_kg_inference.kbN+1.sbert.fbqa &
# python -m 2_kg_inference.kbN+1.bm25.mqa &