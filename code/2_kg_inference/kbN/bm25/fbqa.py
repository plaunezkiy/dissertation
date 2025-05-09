import gc
import os
import torch
import gc
import torch
import pandas as pd
from tqdm import tqdm
from utils.graph import KGraphPreproc
from utils.graph.chain import GraphChain
from utils.llm.mistral import MistralLLM
from utils.prompt import GRAPH_QA_PROMPT, ENTITY_PROMPT
from utils.file import export_results_to_file

device = torch.device("cuda:0")
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
fbqa = pd.read_csv("/datasets/FreebaseQA/FbQA-eval-1000.csv", index_col=0)
fbkb_graph = KGraphPreproc.get_fbkb_graph()

mistral = MistralLLM()
chain = GraphChain.from_llm(
    llm=mistral,
    graph=fbkb_graph,
    qa_prompt=GRAPH_QA_PROMPT,
    entity_prompt=ENTITY_PROMPT,
    verbose=False,
)


for depth in [1,3]:
    # set the depth
    chain.exploration_depth = depth
    chain.ranking_strategy = "bm25"
    # init experiment
    experiment_name = f"kb{depth}"
    res_path = f"/datasets/FreebaseQA/results/{experiment_name}.csv"
    results = []
    id_list = []
    l = 0
    # load if preinit'ed
    if os.path.isfile(res_path):
        r_df = pd.read_csv(res_path)
        l = len(r_df)
        results = list(r_df.Model.values)
    # load q's
    fbqa = pd.read_csv("/datasets/FreebaseQA/FbQA-eval-1000.csv", index_col=0)
    # run through
    for c, (i, r) in enumerate(tqdm(list(fbqa.iterrows()))):
        id_list.append(i)
        if c < l:
            continue
        q, a = get_fbqa_data(r)
        response = get_response(q)
        results.append(response)
        # backup every 10 qs
        if c % 10 == 0:
            export_results_to_file(res_path, results, id_list)
    export_results_to_file(res_path, results, id_list)

# python -m 2_kg_inference.kbN.bm25.fbqa &
# python -m 2_kg_inference.kbN.bm25.mqa &