import networkx as nx


class Attributes(object):
    def __init__(self, visit_count, weight):
        self.visit_count = visit_count
        self.weight = weight

    def __str__(self):
        return 'visit_count=%d\nweight=%d' % (self.visit_count, self.weight)


class GameTree(nx.DiGraph):
    '''Extended networkx DiGraph with functions for move, state combinations and drawing.'''

    def __init__(self, initial_state):
        super().__init__()
        self.attributes = {}

        self.add_node(initial_state)
        self.attributes[initial_state] = Attributes(visit_count=0, weight=0)

    def get_successors(self, state):
        '''return successors of a state as a dictionary {move: state}'''
        return {self.get_edge_data(*edge)['move']: edge[1] for edge in self.edges(state)}

    def add_successor(self, state, move, new_state):
        self.add_node(new_state)
        self.attributes[new_state] = Attributes(visit_count=0, weight=0)
        self.add_edge(state, new_state, move=move)

    def get_ancestors(self, state):
        return nx.ancestors(self, state)

    def draw(self):
        node_labels = {node: str(self.attributes[node]) + '\n\n' + str(node) for node in self.nodes()}
        edge_labels = {edge: self.get_edge_data(*edge)['move'] for edge in self.edges()}
        pos = nx.nx_pydot.graphviz_layout(self, prog='dot')
        nx.draw_networkx(self, pos=pos, labels=node_labels, arrows=False, font_family='monospace')
        nx.draw_networkx_edge_labels(self, pos=pos, edge_labels=edge_labels, font_family='monospace')
