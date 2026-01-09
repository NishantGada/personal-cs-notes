# Algorithm to find all Strongly Connected Components (SCC) in a graph

- A SCC is a component, within a graph, in which every node (of that component) is connected to every other node (in that same component)
- For simpler understanding, I personally think of it as finding a cycle. Not literally, but if can kinda see a cycle in a component of a graph, it probably means that it could be a SCC. Not theoretically proven, just my own understanding for simplification of concept

- Example:
  - This graph has 3 Strongly Connected Components
![example image](SCC%20example.png)

- Only applicable for directed graphs as a undirected graph will always be strongly connected
- Also, we use Topological sorting to find each SCC in a graph, therefore it will HAVE to be a directed graph
