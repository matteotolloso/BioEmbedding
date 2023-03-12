
class Node:

    def __init__(self, function, param):
        self.function = function
        self.param = param

        self.parent = None
        self.childs = []

        self.result = None
    
    def compute(self, previous_stage_output):
        self.result = self.function(previous_stage_output=previous_stage_output, **self.param)
        return self.result

class ExecutionTree:

    def __init__(self, input):
        self.input = input
        self.root = Node(lambda previous_stage_output : previous_stage_output, {})
        self.frontier = [self.root]
    
    def add_stage(self, function, args):
        
        new_frontier = []
        for n in self.frontier:
            new_n = Node(function, args)
            new_n.parent = n
            n.childs.append(new_n)
            new_frontier.append(new_n)
        
        self.frontier = new_frontier

    def add_multistage(self, function, list_args):

        new_frontier = []
        for n in self.frontier:
            for args in list_args:
                new_n = Node(function, args)
                new_n.parent = n
                n.childs.append(new_n)
                new_frontier.append(new_n)
        
        self.frontier = new_frontier
    
    
    def recursive_dfs(self, node, previous_stage_output):
        
        current_output = node.compute(previous_stage_output=previous_stage_output)
        for c in node.childs:
            self.recursive_dfs(c, current_output)


    def compute(self):

        self.recursive_dfs(self.root, self.input)

    def get_args_path(self, node):
        args_path = []
        current_node = node
        while current_node.parent != None:
            args_path.append(str(current_node.function) + " " + str(current_node.param))
            current_node = current_node.parent

        args_path.reverse()

        return args_path
    
    def get_results(self):
        
        results = [ (n.result, self.get_args_path(n)) for n in self.frontier]

        return results
        
        