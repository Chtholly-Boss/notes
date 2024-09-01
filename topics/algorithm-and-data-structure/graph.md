# Graph
## Description
A graph can be described by a **Vertex List** and **Edge List**
```cpp
/**
  * @adjvex: dst vertex in a vertex list
*/
typedef struct node{
  int adjvex;
  node *next;
} EdgeNode;

typedef struct {
  int vex;
  EdgeNode* edgeList;
} Vertex;
// So you can construct a graph using a Vertex List
Vertex g[NumOfVertexs];
// And then add edges to each vertex
```
In python, you can represent a graph using `List`

## Traversing
### Depth-First-Search
```python
# depth-first search
from typing import List
def dfs(graph: List[List[int]], vertex: int, visited: List[bool]):
    # Mark the current vertex as visited
    visited[vertex] = True
    print(vertex)  # Process the current vertex (e.g., print it)

    # Recur for all the vertices adjacent to this vertex
    for neighbor in graph[vertex]:
        if not visited[neighbor]:
            dfs(graph, neighbor, visited)

def dfs_traversal(graph: List[List[int]], n: int):
    visited = [False] * n  # Initialize all vertices as not visited
    for i in range(n):
        if not visited[i]:  # If the vertex is not visited
            dfs(graph, i, visited)
```

The Following Problems may be useful:
* [number-of-island](https://leetcode.cn/problems/number-of-islands/description/)
  * This problem shows that DFS is indeed to search all the adjcent nodes
  * You don't need to convert the matrix to a graph
* [max-area-of-island](https://leetcode.cn/problems/max-area-of-island/submissions/558521767/)
  * This problem shows the usage of DFS return value
### Breadth-First-Search
```python
# breadth first search
from typing import List
from collections import deque

def bfs(graph: List[List[int]], start_vertex: int, visited: List[bool]):
    queue = deque([start_vertex])  # Initialize the queue with the start vertex
    visited[start_vertex] = True  # Mark the start vertex as visited

    while queue:
        vertex = queue.popleft()  # Dequeue a vertex
        print(vertex)  # Process the current vertex (e.g., print it)

        # Recur for all the vertices adjacent to this vertex
        for neighbor in graph[vertex]:
            if not visited[neighbor]:  # If the neighbor hasn't been visited
                visited[neighbor] = True  # Mark neighbor as visited
                queue.append(neighbor)  # Enqueue the neighbor

def bfs_traversal(graph: List[List[int]], n: int):
    visited = [False] * n  # Initialize all vertices as not visited
    for i in range(n):
        if not visited[i]:  # If the vertex is not visited
            bfs(graph, i, visited)
```

* [number-of-island](https://leetcode.cn/problems/number-of-islands/description/)
  * This problem can also be solved using BFS
  * The key in BFS is the **quene**

### Union Find
A Data Structure especially for **Union** and **Find**. 
```python
class UnionFind:
    def __init__(self):
      # a list with each value represents its parent
        self.parent = list(range(26))
    def find(self,index):
        if index == self.parent[index]:
            return index
        # Route compression.
        # each node in the find route will be directly connect to the root
        self.parent[index] = self.find(self.parent[index])
        return self.parent[index]
    def union(self,index1,index2):
        self.parent[self.find(index1)] = self.find(index2)
```

* [number-of-province](https://leetcode.cn/problems/number-of-provinces/description/)
  * This problem is the direct implementation of Union-Find, especially maintain a `branch` variable.
* [Satisfiability-of-equality-equations](https://leetcode.cn/problems/satisfiability-of-equality-equations/)
  * This problem shows that Union-Find is suitable for processing relations.
