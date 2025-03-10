from typing import Any, Dict, Optional
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import BasePromptTemplate
from langchain.chains import GraphQAChain
from langchain.chains.llm import LLMChain
import heapq
from sentence_transformers import SentenceTransformer
from utils.link_prediction import extract_predicted_edges
from utils.prompt import GRAPH_QA_PROMPT, ENTITY_PROMPT, \
    NO_CONTEXT_PROMPT, EVALUATE_CONTEXT_PROMPT, PREDICT_EDGE_PROMPT


class ToGLPChain(GraphQAChain):
    max_exploration_depth: int = 10
    beam_width: int = 25
    max_triplets: int = 250
    plain_qa_chain: LLMChain
    evaluate_context_chain: LLMChain
    link_predictor_llm: LLM
    sbert: SentenceTransformer

    def generate_answer_with_context(self, question, reasoning_chain):
        context = "\n".join([
            ";".join(chain) for chain in (
                ["-".join(triplet) for triplet in triplets] for triplets in reasoning_chain
            )
        ])
        self._run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        self._run_manager.on_text(context, color="green", end="\n", verbose=self.verbose)
        return self.qa_chain(
            {"question": question, "context": context},
            callbacks=self._run_manager.get_child(),
        )

    def predict_n_edges(self, question: str, entities: list, n: int = 5):
        """
        Uses a small LLM to predict N potential edges
        to explore on the graph, then finds the closest
        real match for each using embedding similarity.
        Returns the real edges
        [
            (score, edge, [tails]),
        ]
        """
        # init
        # {"edge": {"tails": [entities], "embedding": []}
        rels = {}
        for entity in entities:
            ent_mid = self.graph.preprocessed_nodes.get(entity, None)
            for h,t, attrs in self.graph._graph.edges(ent_mid, data=True):
                rel = attrs.get("relation", None)
                if rel:
                    rels[rel] = rels.get(rel, {})
                    rels[rel]["embedding"] = attrs.get("embedding", [])
                    rels[rel]["tails"] = rels[rel].get("tails", set())
                    rels[rel]["tails"].add(t)
        # predict
        prompt = PREDICT_EDGE_PROMPT.format(
            question=question, entities=entities, no_items=n
        )
        predicted_response = self.link_predictor_llm(prompt)
        predicted_edge_candidates = extract_predicted_edges(predicted_response)
        edges = []
        # match and score
        for edge_candidate in predicted_edge_candidates:
            candidate_embedding = self.sbert.encode(edge_candidate)
            scored_edges = [
                (
                    self.sbert.similarity(
                        candidate_embedding, 
                        rels[edge]["embedding"]
                    )[0].cpu().numpy()[0],
                    edge,
                    rels[edge]["tails"]
                ) for edge in rels.keys()
            ]
            scored_edges.sort(key=lambda t: t[0], reverse=True)
            edges.append(scored_edges[0])
        return edges[:n+1]

    def local_connectivity_rerank(self, P, entity, edge, edge_score, candidates_tails):
        """
        [(score, triplet),]
        """
        # go through all candidate tails
        # tail_score = (
        #  1 if link(path_node, tail)
        # ) for path_node in P
        scored_triplets = []
        for tail in candidates_tails:
            if self.graph.mid2name.get(tail) is None:
                continue
            tail_score = 0
            for step in P:
                tentative_tail_score = sum([
                    # 0.5 if connected to head/tail, 1 if to both, 0 if to none
                    0.5 if self.graph._graph.has_edge(
                        self.graph.name2mid.get(triplet[-1]),
                        tail
                    ) else 0 +
                    0.5 if self.graph._graph.has_edge(
                        self.graph.name2mid.get(triplet[0]),
                        tail
                    ) else 0
                    for triplet in step
                ]) / len(step)
                tail_score += tentative_tail_score
            scored_triplets.append(
                (
                    tail_score/len(P) * edge_score,
                    (entity, edge, self.graph.mid2name.get(tail))
                )
            )
        scored_triplets.sort(key=lambda t: t[0], reverse=True)
        return scored_triplets

    def check_enough_context_to_answer(self, question, reasoning_chain):
        context = "\n".join([
            ";".join(chain) for chain in (
                ["-".join(triplet) for triplet in triplets] for triplets in reasoning_chain
            )
        ])
        self._run_manager.on_text("Checking if the context is sufficient:", end="\n", verbose=self.verbose)
        self._run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        self._run_manager.on_text(context, color="green", end="\n", verbose=self.verbose)
        answer = self.evaluate_context_chain(
            {"question": question, "context": context},
            callbacks=self._run_manager.get_child(),
        )
        self._run_manager.on_text(f"Model decision: {answer['text']}", end="\n", verbose=self.verbose)
        if "yes" in answer["text"].strip().lower():
            return True
        return False

    def tog_answer_predict_link(self, question, topic_entities, max_depth=4, beam_width=25):
        """
        Performs ToG with link prediction and tail reranking
        instead of LLM-based scoring.
        """
        # init the entities
        # print("Topic entities", topic_entities)
        explored_entities = set(topic_entities)
        # print("Explroed Entities", explored_entities)
        P = [
            [
                (e,) for e in explored_entities
            ]
            # step: [triplet1, ...], 
        ] # max_depth X beam_width
        # print("P", P)
        # iteratively traverse
        d_count = 0
        for depth in range(1, max_depth):
            d_count += 1
            tail_entities = set()
            for t in P[depth-1]:
                tail_entities.add(t[-1])
            # get the unvisited ones
            unexplored_entities = tail_entities.difference(explored_entities)
            # add them as visited (for next iteration)
            explored_entities.update(unexplored_entities)

            edge_heap = []
            # predict and rerank edges
            for entity in tail_entities:
                entity_mid = self.graph.preprocessed_nodes.get(entity, None)
                if entity is None:
                    continue
                # ??? potentially get all prev entities for that chain
                # should return real edges (match inside call)
                predicted_edges = self.predict_n_edges(
                    question, [entity], #n=5
                )
                # print("Predited edges", [(t[0], t[1]) for t in predicted_edges])
                # Add and rebalance edge heap
                for score, edge, tails in predicted_edges:
                    heapq.heappush(edge_heap, (score, entity, edge, tails))
            # predict and rank tail entities
            triplet_heap = []
            for edge_score, entity, edge, candidates_tails in heapq.nlargest(len(topic_entities), edge_heap):
                # rerank based on a scoring function
                # local connectivity:
                # print(entity, edge, candidates_tails)
                # print()
                top_triplets = self.local_connectivity_rerank(P, entity, edge, edge_score, candidates_tails)
                # add and rebalance triplet heap
                for (score, triplet) in top_triplets:
                    heapq.heappush(triplet_heap, (score, triplet))
            top_new_triplets = sorted(triplet_heap, key=lambda x: -x[0])[:beam_width]
            # print("Top new triplets", top_new_triplets)
            # construct paths
            P.append([t[1] for t in top_new_triplets])
            # print("P:", P)
            # Reasoning
            is_enough = self.check_enough_context_to_answer(question, reasoning_chain=P)
            if is_enough:
                break
            else:
                continue
        return {
            "text": self.generate_answer_with_context(question, reasoning_chain=P),
            "depth": d_count
            }
    
    #### INITIALIZATION AND INVOKATION

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Extract entities, look up info and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        self._run_manager = _run_manager
        question = inputs[self.input_key]
        topic_entities = inputs.get("topic_entities", [])
        # 
        result = self.tog_answer_predict_link(question, topic_entities)
        return {self.output_key: result[self.qa_chain.output_key], "depth": result["depth"]}
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        link_predictor_llm: LLM,
        qa_prompt: BasePromptTemplate = GRAPH_QA_PROMPT,
        qa_no_context_prompt: BasePromptTemplate = NO_CONTEXT_PROMPT,
        evaluate_context_prompt: BasePromptTemplate = EVALUATE_CONTEXT_PROMPT,
        entity_prompt: BasePromptTemplate = ENTITY_PROMPT,
        sbert: SentenceTransformer=None,
        **kwargs: Any,
    ) -> GraphQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        plain_qa_chain = LLMChain(llm=llm, prompt=qa_no_context_prompt)
        evaluate_context_chain = LLMChain(llm=llm, prompt=evaluate_context_prompt)
        entity_chain = LLMChain(llm=llm, prompt=entity_prompt)
        # embed_sbert = lambda q: model.encode(q)

        return cls(
            qa_chain=qa_chain,
            plain_qa_chain=plain_qa_chain,
            evaluate_context_chain=evaluate_context_chain,
            entity_extraction_chain=entity_chain,
            link_predictor_llm=link_predictor_llm,
            sbert=sbert,
            **kwargs,
        )
