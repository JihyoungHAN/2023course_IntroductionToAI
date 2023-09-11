from tiles import TilesNode
from queue import PriorityQueue


def heuristic(node: TilesNode) -> int:
    num = 0
    value = 0
    for i in range(4):
        for j in range(4):
            num = num + 1
            if int(node.state[i][j]) != num: 
                value = 1 + value
    return value 
            
    """
    Evaluate the heuristic value of the current node.
    This implementation simply counts the number of misplaced tiles.

    Returns
    -------
    heuristic_value : int
        The heuristic value of the current node.
    """


def AStar(root, heuristic: callable) -> TilesNode or None:
    unexplored = PriorityQueue()
    counter = 0
    unexplored.put((0, counter, root))
    # HINT: PriorityQueue.put() takes a tuple as input
    # To sort the queue items, it uses the first element of each tuple
    # If the first elements are equal, it uses the second element, and so on
    # You may implement a counter to resolve ties
    explored = set()
    g_score = {root: 0}
    f_score = {root: heuristic(root)}
    while not unexplored.empty():
        children = root.get_children()
        for i in children:
            g_score[i] = 1
            f_score[i] = heuristic(i)   
        min_node = find_min_value_key(f_score)
        del f_score[min_node]
        tuple_state = tuple([tuple(i) for i in root.state])
        explored.add(tuple_state)
        root.state = min_node.state
        if heuristic(root) == 1: 
            break
    print(root.get_path())
    return None  # return None if no path was found

def find_min_value_key(dictionary):
    min_value = min(dictionary.values())
    min_value_key = None
    for key, value in dictionary.items():
        if value == min_value:
            min_value_key = key
            break
    return min_value_key

def accept(self):
    tile = []
    for i in range(0, 4):
        temp = input().split(" ")
        temp_int = [int(x) for x in temp]
        tile.append(temp_int) 
    return tile 

def main(): 
    print("Enter the start tile node \n")
    root= [] 
    root = accept(root)
    root = TilesNode(root)
    solution = AStar(root, heuristic)

if __name__ == "__main__":
    main()