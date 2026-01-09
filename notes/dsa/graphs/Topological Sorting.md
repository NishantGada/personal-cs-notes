# Topological sorting

#### Only applicable on DAGs – Directed Acyclic Graphs

#### Two primary ways to get Topological Sorting:
- DFS (regular topo sort)
  - Typical DFS algorithm with a minor change – use a stack and after processing all neighbors for a node, add the curr node to a stack
  - To get the final topological sort, simply pop and print all the stack values 


- BFS (Kahn’s algorithm)
  - Uses Queue to store sorted order
  - Primary logic is to find the indegree for each node where indegree is the number of incoming edges
  - Start with the nodes which have an indegree = 0 and process their neighbors
  - For each neighbor that you process, reduce that neighbors indegree by 1 and if there’s any neighbor node whose indegree becomes 0 after the reduction, add that to the queue