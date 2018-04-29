import numpy as np
import networkx as nx

from .attributes import StateAttributes, TransitionAttributes


class ActionAlreadyExistsException(Exception):
    pass


class Tree(nx.DiGraph):
    def __init__(self, root):
        super().__init__()
        self.add_node(root, attributes=StateAttributes())

    @property
    def states(self):
        return self.nodes

    def add_state(self, state, source, action):
        if action in self.get_actions(source):
            raise ActionAlreadyExistsException()
        self.add_node(state, attributes=StateAttributes())
        self.add_edge(source, state, action=action, attributes=TransitionAttributes())

    def get_actions(self, source):
        return set([self.get_edge_data(*edge)['action'] for edge in self.edges(source)])

    def get_successor(self, source, action):
        successor, = [edge[1] for edge in self.edges(source) if self.get_edge_data(*edge)['action'] == action]
        return successor

    def get_attributes(self, source, action=None):
        if action is None:
            return self._get_state_attributes(source)
        else:
            return self._get_transition_attributes(source, action)

    def _get_state_attributes(self, source):
        return self.nodes[source]['attributes']

    def _get_transition_attributes(self, source, action):
        transition, = [edge for edge in self.edges(source) if self.get_edge_data(*edge)['action'] == action]
        return self.get_edge_data(*transition)['attributes']

    def get_path_to_root(self, source):
        return nx.ancestors(self, source) | {source}

    def draw(self):
        node_labels = {node: self._get_node_label(node) for node in self.nodes()}
        edge_labels = {edge: self._get_edge_label(edge) for edge in self.edges()}
        pos = nx.nx_pydot.graphviz_layout(self, prog='dot')
        nx.draw_networkx(self, pos=pos, labels=node_labels, arrows=False, font_family='monospace', font_size=8)
        nx.draw_networkx_edge_labels(self, pos=pos, edge_labels=edge_labels, font_family='monospace', font_size=8)

    def _get_node_label(self, node):
        return str(self.get_attributes(node)) + '\n\n' + str(node)

    def _get_edge_label(self, edge):
        action = self.get_edge_data(*edge)['action']
        return str(action) + '\n\n' + str(self.get_attributes(edge[0], action))
