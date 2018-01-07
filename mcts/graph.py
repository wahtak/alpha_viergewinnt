import networkx as nx


class MCTSGraph(nx.DiGraph):
    '''Extended networkx DiGraph with functions for move, state combinations and drawing.'''

    def __init__(self, initial_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_node(initial_state)

    def get_successors(self, state):
        '''return successors of a state as a dictionary {move: state}'''
        return {self.get_edge_data(*edge)['move']: edge[1] for edge in self.edges(state)}

    def add_successor(self, state, move, new_state):
        self.add_edge(state, new_state, move=move)

    def draw(self):
        node_labels = {node: str(node) for node in self.nodes()}
        edge_labels = {edge: self.get_edge_data(*edge)['move'] for edge in self.edges()}
        pos = nx.nx_pydot.graphviz_layout(self, prog='dot')
        nx.draw_networkx(self, pos=pos, labels=node_labels, arrows=False, font_family='monospace')
        nx.draw_networkx_edge_labels(self, pos=pos, edge_labels=edge_labels)