from langchain_community.graphs.networkx_graph import NetworkxEntityGraph, KnowledgeTriple
from typing import List
import networkx as nx
from tqdm import tqdm
from utils.preprocessing import preprocess_text
import pandas as pd
import ast
import os
import csv
import re

class RankedTriplet:
    """
    Instance of a ranked triplet in a IR system
    """
    def __init__(self, subject, relation, object, relevance_score):
        self.subject = subject
        self.relation = relation
        self.object = object
        self.score = float(relevance_score)
    
    def __repr__(self):
        return f"{self.score}"
    
    def __str__(self):
        return f"{self.subject} {self.relation} {self.object}"

    def __lt__(self, other):
        return self.rank < other.rank


class KGraphPreproc(NetworkxEntityGraph):
    def generate_preprocessed_nodes(self) -> None:
        # {proc_node: node_id}
        self.preprocessed_nodes = {}
        for node in self._graph.nodes:
            proc_node = preprocess_text(self.mid2name.get(node, ""))
            self.preprocessed_nodes[proc_node] = node
    
    def get_entity_knowledge(self, entity: str, depth: int=1) -> List[str]:
        """
        Custom implementation of entity search and knowledge extraction,
        instead of exact match, does preprocessing and then matches
        """
        proc_node = preprocess_text(entity)
        g_entity = self.preprocessed_nodes.get(proc_node, None)
        return super().get_entity_knowledge(g_entity, depth)
    
    def _get_entity_knowledge(self, entity: str, depth: int = 1) -> List[str]:
        """Get information about an entity."""
        import networkx as nx

        # TODO: Have more information-specific retrieval methods
        if not self._graph.has_node(entity):
            return []

        results = []
        for src, sink in nx.dfs_edges(self._graph, entity, depth_limit=depth):
            relation = self._graph[src][sink]["relation"]
            results.append(f"{self.mid2name[src]} {relation} {self.mid2name[sink]}")
        return results
    
    def get_subgraph(self, entity: str, depth: int=1) -> NetworkxEntityGraph:
        nodes_within_depth = nx.single_source_shortest_path_length(self._graph, entity, cutoff=depth).keys()
        # Create the subgraph containing only the nodes within the specified depth
        subgraph = self._graph.subgraph(nodes_within_depth).copy()
        return subgraph
    
    def embed_triplets(self, embedding_function, cache_path):
        # check cache
        # print("Checking if already embedded")
        if "embedding" in list(self._graph.edges(data=True))[0][2]:
            # print("Already embedded")
            return
        print("Checking embedding cache")
        if os.path.isfile(cache_path):
            print("Loading embedding cache")
            with open(cache_path, "r") as cache_file:
                reader = csv.reader(cache_file)
                # skip header
                next(reader)
                for (key, embedding) in tqdm(reader):
                    embedding = re.sub(r"(\d)\s+", r"\1, ", embedding)
                    # print(embedding)
                    self._graph.edges[eval(key)]["embedding"] = eval(embedding)
                return
        # embedding
        print("Computing embeddedings")
        for u,v in tqdm(self._graph.edges()):
            if "embedding" in self._graph.edges[u, v]:
                continue
            head = self.mid2name.get(u, None)
            tail = self.mid2name.get(v, None)
            rel = self._graph.edges[u, v].get("relation", None)
            triplet = f"{head} {rel} {tail}"
            self._graph.edges[u, v]["embedding"] = embedding_function(triplet)
        # caching the result
        print("Caching the computed embeddings")
        with open(cache_path, "w") as cache_file:
            writer = csv.writer(cache_file)
            writer.writerow(["key", "embedding"])
            for u,v in tqdm(self._graph.edges()):
                embedding = self._graph.edges[u, v].get("embedding", None)
                writer.writerow([(u,v), embedding])

    @classmethod
    def get_fbkb_graph(cls):
        mid2name = pd.read_csv("/datasets/FB15k-237/mid2name.txt", sep="\t", header=None)
        mid2name_dict = dict(zip(mid2name[0], mid2name[1]))
        name2mid_dict = dict(zip(mid2name[1], mid2name[0]))
        # 
        fbkb_graph = cls()
        fbkb_graph.mid2name = mid2name_dict
        fbkb_graph.name2mid = name2mid_dict
        # 
        fbkb = pd.DataFrame()
        for split_path in [
                "/datasets/FB15k-237/kb_dump.csv",
                *[f"/datasets/FB15k-237/{split}.txt" for split in ["train", "valid", "test"]]
            ]:
            split_df = pd.read_csv(split_path, sep="\t", header=None)
            fbkb = pd.concat([fbkb, split_df])
        fbkb.rename(columns={0: "subject", 1: "relation", 2: "object"}, inplace=True)
        # 
        for i, r in fbkb.iterrows():
            # s = mid2name_dict.get(r.subject, None)
            s = r.subject
            o = r.object
            # o = mid2name_dict.get(r.object, None)
            if s and o:
                triplet = KnowledgeTriple(
                    s,
                    r.relation,
                    o
                )
                fbkb_graph.add_triple(triplet)
        # 
        fbkb_graph.generate_preprocessed_nodes()
        fbkb_graph._graph = fbkb_graph._graph.to_undirected()
        return fbkb_graph
    
    @classmethod
    def get_metaqa_graph(cls):
        mid2name = pd.read_csv("/datasets/MetaQA/KB/kb_entity_dict.txt", sep="\t", header=None)
        mid2name_dict = dict(zip(mid2name[0], mid2name[1]))
        name2mid_dict = dict(zip(mid2name[1], mid2name[0]))
        
        metaqa_kb = pd.read_csv("/datasets/MetaQA/KB/kb.txt", sep="|", header=None)
        metaqa_kb.rename(columns={0: "subject", 1: "relation", 2: "object"}, inplace=True)
        
        # construct the KG
        mqa_graph = cls()
        mqa_graph.mid2name = mid2name_dict
        mqa_graph.name2mid = name2mid_dict
        for i, r in metaqa_kb.iterrows():
            triplet = KnowledgeTriple(
                name2mid_dict.get(r.subject, None),
                r.relation,
                name2mid_dict.get(r.object, None),
            )
            mqa_graph.add_triple(triplet)
        # prep the graph
        mqa_graph.generate_preprocessed_nodes()
        # account for directional edges (llm will figure it out)
        mqa_graph._graph = mqa_graph._graph.to_undirected()
        return mqa_graph
