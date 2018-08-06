class Graph(object):

    def __init__(self):
        self.placeholders = []
        self.random_variables = []
        self.operations = []
        self.constants = []

    def reset(self):
        for node in self.placeholders:
            node.visited = False
        for node in self.random_variables:
            node.visited = False
        for node in self.operations:
            node.visited = False
        for node in self.constants:
            node.visited = False

class Node(object):

    def __init__(self, name):
        self.name = name
        self.visited = False

        # The set of nodes which depends on the computation of this
        # one
        self.consumers = []


