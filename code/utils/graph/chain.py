    from typing import Any, Dict, Iterator, List, Mapping, Optional, Literal
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_community.graphs.networkx_graph import get_entities
from langchain.chains import GraphQAChain
import bm25s
import networkx as nx
import heapq
from utils.preprocessing import stemmer, preprocess_text
from sentence_transformers import SentenceTransformer

# class RankedTriplet:
#     def __init__(self, head, rel, tail, query):
#         self.triplet = f"{mid2name_dict.get(head, "") {mid2name_dict.get(rel, "")} {mid2name_dict.get(tail, "")}}"
#         self.score = 


class GraphChain(GraphQAChain):
    exploration_depth: int = 4
    max_triplets: int = 250
    ranking_strategy: Literal["bm25", "sbert"] = "sbert"
    sbert_cache_path: str = "/datasets/FB15k-237/cache/sbert.csv"

    #### GRAPH OPERATIONS AND RANKING
    def extract_triplets(self, entities, depth=1):
        subgraph_entities = set()
        for entity in entities:
            processed_entity = preprocess_text(entity)
            node = self.graph.preprocessed_nodes.get(processed_entity)
            if node is None:
                continue
            bfs_nodes = nx.single_source_shortest_path_length(
                self.graph._graph,
                node,
                cutoff=depth
            ).keys()
            subgraph_entities.update(bfs_nodes)
        subgraph = self.graph._graph.subgraph(subgraph_entities)
        # 
        triplets = []
        mid2name_dict = self.graph.mid2name
        for head, tail, attrs in subgraph.edges(data=True):
            rel = attrs.get("relation", None)
            if head and rel and tail:
                triplets.append(
                    (f"{mid2name_dict.get(head, '')} {rel} {mid2name_dict.get(tail, '')}", attrs.get("embedding", []))
                )
        return triplets
    
    def rerank_triplets_bm25(self, question, triplets, top_k=max_triplets):
        top_triplets = []
        # BM25 ranking
        if triplets:
            triplet_tokens = bm25s.tokenize(
                map(
                    preprocess_text, 
                    # extract text from (text, embedding)
                    map(lambda t: t[0], triplets)
                ),
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
            # !!! limit
            top_triplets = top_triplets[0][:top_k]
        return top_triplets
    
    def rerank_triplets_sbert(self, question, triplets, embedding_function, similarity_function, top_k=max_triplets):
        top_triplets = []        
        # BM25 ranking
        if triplets:
            question_embedding = embedding_function(question)
            min_heap = []
            for idx, triplet_data in enumerate(triplets):
                triplet_embedding = triplet_data[1]
                score = similarity_function(question_embedding, triplet_embedding)
                # Push to heap if we haven't reached N elements yet
                if len(min_heap) < top_k:
                    heapq.heappush(min_heap, (score, idx))
                else:
                    # Replace the smallest element if current score is larger
                    if score > min_heap[0][0]:
                        heapq.heappushpop(min_heap, (score, idx))
            top_n = sorted(min_heap, key=lambda x: -x[0])
            top_triplets = [triplets[idx] for (score, idx) in top_n]
        return top_triplets
    

    #### INITIALIZATION AND INVOKATION

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

        if self.ranking_strategy == "sbert":
            _run_manager.on_text("Embedding the graph", color="green", end="\n", verbose=self.verbose)
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embed_sbert = lambda q: model.encode(q)
            # self.graph.embed_triplets(embedding_function=embed_sbert, cache_path=self.sbert_cache_path)

        # extract triplets
        triplets = self.extract_triplets(entities, depth=self.exploration_depth)
        
        top_triplets = []
        # rerank triplets
        if self.ranking_strategy == "bm25":
            top_triplets = self.rerank_triplets_bm25(question, triplets, top_k=self.max_triplets)
        elif self.ranking_strategy == "sbert":
            # EXPECTING THE GRAPH TO BE EMBEDDED
            top_triplets = self.rerank_triplets_sbert(
                question, triplets, top_k=self.max_triplets, 
                embedding_function=embed_sbert,
                similarity_function=lambda u, v: model.similarity(u, v)
            )
        # combine and generate answer
        context = "\n".join(
            map(lambda t: t[0], top_triplets)
        )

        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(context, color="green", end="\n", verbose=self.verbose)
        result = self.qa_chain(
            {"question": question, "context": context},
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: result[self.qa_chain.output_key]}