from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_community.graphs.networkx_graph import get_entities
from langchain.chains import GraphQAChain


class GraphChain(GraphQAChain):
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
        for entity in entities:
            # introduce preprocessing
            processed_entity = preprocess_text(entity)
            entity_mid = name2mid.get(processed_entity, None)
            # continue
            triplets = self.graph.get_entity_knowledge(entity_mid)
            for triplet in triplets:
                t = triplet.split()
                all_triplets.append(" ".join([mid2name_dict.get(t[0], ""), t[1], mid2name_dict.get(t[2], "")]))
        # limit to 600
        context = "\n".join(all_triplets[:600])
        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(context, color="green", end="\n", verbose=self.verbose)
        result = self.qa_chain(
            {"question": question, "context": context},
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: result[self.qa_chain.output_key]}