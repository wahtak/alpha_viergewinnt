import networkx as nx

from .attributes import StateAttributes, TransitionAttributes


class ActionAlreadyExistsException(Exception):
    pass


class GameStateGraph(nx.DiGraph):
    def __init__(self, root):
        super().__init__()
        self.add_node(root, attributes=StateAttributes())

    @property
    def states(self):
        return self.nodes

    def add_successor(self, successor, source, action):
        if action in self.get_actions(source):
            raise ActionAlreadyExistsException()
        self.add_node(successor, attributes=StateAttributes())
        self.add_edge(source, successor, action=action, attributes=TransitionAttributes())

    def get_actions(self, source):
        return [self.get_edge_data(*edge)['action'] for edge in self.edges(source)]

    def get_successor(self, source, action):
        successor, = [edge[1] for edge in self.edges(source) if self.get_edge_data(*edge)['action'] == action]
        return successor

    def has_successors(self, source):
        return len(self.edges(source)) > 0

    def get_state_attributes(self, state):
        return self.nodes[state]['attributes']

    def get_transition_attributes(self, source, action):
        transition, = [edge for edge in self.edges(source) if self.get_edge_data(*edge)['action'] == action]
        return self.get_edge_data(*transition)['attributes']

    def get_predecessors(self, state):
        return set(self.predecessors(state))

    def draw(self):
        state_labels = {node: self._get_state_label(node) for node in self.nodes()}
        transition_labels = {edge: self._get_transition_label(edge) for edge in self.edges()}
        pos = nx.nx_pydot.graphviz_layout(self, prog='dot')
        nx.draw_networkx(self, pos=pos, labels=state_labels, arrows=False, font_family='monospace', font_size=8)
        nx.draw_networkx_edge_labels(self, pos=pos, edge_labels=transition_labels, font_family='monospace', font_size=8)

    def _get_state_label(self, node):
        return str(self.get_state_attributes(node)) + '\n\n' + str(node)

    def _get_transition_label(self, edge):
        action = self.get_edge_data(*edge)['action']
        return str(action) + '\n\n' + str(self.get_transition_attributes(edge[0], action))
