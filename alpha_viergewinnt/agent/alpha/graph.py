import numpy as np
import networkx as nx


class ActionAlreadyExistsException(Exception):
    pass


class GameStateGraph(nx.DiGraph):
    def __init__(self, root=None):
        super().__init__()
        self.root = root
        self.add_node(root, attributes=None)

    @classmethod
    def create_path(cls, root):
        return GameStatePath(root)

    def reset_root(self, root):
        self.root = root
        if root in self.nodes:
            nodes_for_pruning = set(self.nodes) - nx.descendants(self, root) - {root}
            self.remove_nodes_from(nodes_for_pruning)
            self.set_attributes(None, state=root)
        else:
            self.clear()
            self.add_node(root, attributes=None)

    @property
    def states(self):
        return self.nodes

    def add_successor(self, successor, source, action):
        if action in self.get_actions(source):
            raise ActionAlreadyExistsException()
        self.add_node(successor, attributes=None)
        self.add_edge(source, successor, action=action)

    def get_actions(self, source):
        return [self.get_edge_data(*edge)['action'] for edge in self.edges(source)]

    def get_successor(self, source, action):
        successor, = [edge[1] for edge in self.edges(source) if self.get_edge_data(*edge)['action'] == action]
        return successor

    def has_successors(self, source):
        return len(self.edges(source)) > 0

    def get_attributes(self, state):
        return self.nodes[state]['attributes']

    def set_attributes(self, attributes, state):
        self.nodes[state]['attributes'] = attributes

    def get_predecessors(self, state):
        return set(self.predecessors(state))

    def get_mean_node_depth(self):
        node_depths = self._get_node_depths(self.root, 0)
        return np.mean(list(node_depths.values()))

    def _get_node_depths(self, source, start_depth):
        node_depths = {source: start_depth}
        for _, successor in self.edges(source):
            node_depths.update(self._get_node_depths(successor, start_depth + 1))
        return node_depths

    def draw(self):
        visible = self.subgraph({node for node in self.nodes if self.get_attributes(node) is not None})
        state_labels = {node: self._get_state_label(node) for node in visible.nodes()}
        action_labels = {edge: self._get_action_label(edge) for edge in visible.edges()}
        pos = nx.nx_pydot.graphviz_layout(visible, prog='dot')
        nx.draw_networkx(visible, pos=pos, labels=state_labels, arrows=False, font_family='monospace', font_size=8)
        nx.draw_networkx_edge_labels(visible, pos=pos, edge_labels=action_labels, font_family='monospace', font_size=8)
        import matplotlib.pyplot as plt
        plt.show()

    def _get_state_label(self, node):
        return str(self.get_attributes(node)) + '\n\n' + str(node)

    def _get_action_label(self, edge):
        return str(self.get_edge_data(*edge)['action'])


class GameStatePath(GameStateGraph):
    def __init__(self, root):
        super().__init__(root)
        self.leaf = root

    def add_successor(self, successor, action):
        self.add_node(successor)
        self.add_edge(self.leaf, successor, action=action)
        self.leaf = successor

    def get_action(self, source):
        action, = self.get_actions(source)
        return action

    def get_predecessor(self, state):
        predecessor, = self.get_predecessors(state)
        return predecessor
