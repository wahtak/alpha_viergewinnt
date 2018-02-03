import networkx as nx


class TransitionAlreadyExistsException(Exception):
    pass


class Attributes(object):
    def __init__(self, visit_count, weight):
        self.visit_count = visit_count
        self.weight = weight

    def __str__(self):
        return 'visit_count=%d\nweight=%d' % (self.visit_count, self.weight)


class Tree(nx.DiGraph):
    def __init__(self, root):
        super().__init__()
        self.attributes = {}

        self.add_node(root)
        self.attributes[root] = Attributes(visit_count=0, weight=0)

    def get_transitions(self, source):
        return {self.get_edge_data(*edge)['transition'] for edge in self.edges(source)}

    def get_successor(self, source, transition):
        successor, = [edge[1] for edge in self.edges(source) if self.get_edge_data(*edge)['transition'] == transition]
        return successor

    def add_successor(self, source, transition, successor):
        if transition in self.get_transitions(source):
            raise TransitionAlreadyExistsException()
        self.add_node(successor)
        self.attributes[successor] = Attributes(visit_count=0, weight=0)
        self.add_edge(source, successor, transition=transition)

    def get_path_to_root(self, source):
        return nx.ancestors(self, source) | {source}

    def draw(self):
        node_labels = {node: str(self.attributes[node]) + '\n\n' + str(node) for node in self.nodes()}
        edge_labels = {edge: self.get_edge_data(*edge)['transition'] for edge in self.edges()}
        pos = nx.nx_pydot.graphviz_layout(self, prog='dot')
        nx.draw_networkx(self, pos=pos, labels=node_labels, arrows=False, font_family='monospace')
        nx.draw_networkx_edge_labels(self, pos=pos, edge_labels=edge_labels, font_family='monospace')
