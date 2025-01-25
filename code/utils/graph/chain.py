from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_community.graphs.networkx_graph import get_entities
from langchain.chains import GraphQAChain
import bm25s
import networkx as nx
from utils.preprocessing import stemmer, preprocess_text

# class RankedTriplet:
#     def __init__(self, head, rel, tail, query):
#         self.triplet = f"{mid2name_dict.get(head, "") {mid2name_dict.get(rel, "")} {mid2name_dict.get(tail, "")}}"
#         self.score = 


class GraphChain(GraphQAChain):
    exploration_depth: int = 4
    max_triplets: int = 250

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Extract entities, look up info and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        entity_string = self.entity_extraction_chain.run(question)

        _run_manager.on_text("Entities Extracted:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            entity_string, color="green", end="\n", verbose=self.verbose
        )
        entities = get_entities(entity_string)
        context = ""
        all_triplets = []
        
        # 
        subgraph_entities = set()
        for entity in entities:
            processed_entity = preprocess_text(entity)
            node = self.graph.preprocessed_nodes.get(processed_entity)
            if node is None:
                continue
            bfs_nodes = nx.single_source_shortest_path_length(
                self.graph._graph,
                node,
                cutoff=self.exploration_depth
            ).keys()
            subgraph_entities.update(bfs_nodes)
        subgraph = self.graph._graph.subgraph(subgraph_entities)
        # 
        triplets = []
        mid2name_dict = self.graph.mid2name
        for head, tail, attrs in subgraph.edges(data=True):
            rel = attrs.get("relation", None)
            triplets.append(
                f"{mid2name_dict.get(head, '')} {rel} {mid2name_dict.get(tail, '')}"
            )
        top_triplets = []
        # BM25 ranking
        if triplets:
            triplet_tokens = bm25s.tokenize(
                map(preprocess_text, triplets),
                stemmer=stemmer)
            retriever = bm25s.BM25()
            # index
            retriever.index(triplet_tokens)
            # query
            question_tokens = bm25s.tokenize(
                preprocess_text(question),
                stemmer=stemmer
            )
            top_triplets, score = retriever.retrieve(
                question_tokens,
                k=len(triplets),
                corpus=triplets,
                # return_as="tuple",
                sorted=True,
            )
            # !!! limit to 600
            top_triplets = top_triplets[0][:self.max_triplets]
        # print(len(top_triplets))
        # print(top_triplets)
        # print(score[0])

        # 
        # for entity in entities:
        #     # introduce preprocessing
        #     processed_entity = preprocess_text(entity)
        #     entity_mid = name2mid.get(processed_entity, None)
        #     # continue
        #     # extract triplets on the subgraph of depth N
        #     triplets = self.graph.get_entity_knowledge(entity_mid)
        #     for triplet in triplets:
        #         t = triplet.split()
        #         all_triplets.append(" ".join([mid2name_dict.get(t[0], ""), t[1], mid2name_dict.get(t[2], "")]))
        
        context = "\n".join(top_triplets)

        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(context, color="green", end="\n", verbose=self.verbose)
        result = self.qa_chain(
            {"question": question, "context": context},
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: result[self.qa_chain.output_key]}