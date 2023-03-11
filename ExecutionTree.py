
class Node:

    def __init__(self, function, param):
        self.function = function
        self.param = param

        self.parent = None
        self.childs = []
    
    def compute(self, input):
        return self.function(input, **self.param)

class ExecutionTree:

    def __init__(self, input):
        self.input = input
        self.frontier = []
        self.root = Node(lambda x:x, {})
    
    def add_stage(self, function, args):
        
        new_frontier = []
        for n in self.frontier:
            new_n = Node(function, args)
            new_n.parent = self
            n.childs.append(new_n)
            new_frontier.append(new_n)
        
        self.frontier = new_frontier

    def add_multistage(self, function, list_args):

        new_frontier = []
        for n in self.frontier:
            for args in list_args:
                new_n = Node(function, args)
                new_n.parent = self
                n.childs.append(new_n)
                new_frontier.append(new_n)
        
        self.frontier = new_frontier
    
    
    def recursive_dfs(self, node, parent_output):
        
        current_output = node.compute(parent_output)
        for c in node.childs:
            self.recursive_dfs(c, current_output)


    def compute(self):

        self.recursive_dfs(self.root, self.input)
        
        
if __name__ == "__main__":
    et = ExecutionTree("ciao")
    

    
    et.add_stage(lambda x: x+"+", {})
    et.add_stage(lambda x: print(x), {})

    et.compute()