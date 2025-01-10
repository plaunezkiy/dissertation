from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from typing import List
import networkx as nx
from utils.preprocessing import preprocess_text


class KGraphPreproc(NetworkxEntityGraph):
    def generate_preprocessed_nodes(self) -> None:
        # {proc_node: node_id}
        self.preprocessed_nodes = {}
        for node in self._graph.nodes:
            proc_node = " ".join(preprocess_text(node))
            self.preprocessed_nodes[proc_node] = node
    
    def get_entity_knowledge(self, entity: str, depth: int=1) -> List[str]:
        """
        Custom implementation of entity search and knowledge extraction,
        instead of exact match, does preprocessing and then matches
        """
        proc_node = " ".join(preprocess_text(entity))
        g_entity = self.preprocessed_nodes.get(proc_node, None)
        return super().get_entity_knowledge(g_entity, depth)
    
    def get_subgraph(self, entity: str, depth: int=1) -> NetworkxEntityGraph:
        nodes_within_depth = nx.single_source_shortest_path_length(self._graph, entity, cutoff=depth).keys()
        # Create the subgraph containing only the nodes within the specified depth
        subgraph = self._graph.subgraph(nodes_within_depth).copy()
        return subgraph
