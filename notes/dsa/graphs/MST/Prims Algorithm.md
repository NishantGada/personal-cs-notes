# Prim's Algorithm

- According to Prim's Algorithm, we maintain 2 different data structures. 1 for holding nodes that aren't a part of our MST and 1 to hold nodes that are a part of our MST. 

- For the data structure holding non-MST nodes, we use a Priority Queue
  - A Priority Queue (minimum) is used as for this algorithm because we need a data structure that gives us the minimum edge weight between two nodes
  - When starting from an origin node, we consider each of its neighbors and store the distance between the origin and its neighbors in the form of Pairs (int node, int cost), in the PQ
  - This way, whenever we remove/poll a pair from the PQ, we will get the one with the shortest edge weight/cost/distance from the origin

- For the data structure holding MST nodes, we use a boolean Visited array
  - A boolean Visited array is used because whenever we visit a node from the origin, it means that it has the shortest cost/weight/distance from the origin and is eligible to part of the MST
  - We mark the node as visited in our boolean array, which essentially adds that node to our MST

- One variation of MST problems includes calculating the total edge weight/cost of a MST
  - To get this final cost/MST cost, we can maintain a finalCost variable (intialized with 0)
  - Now whenever we remove/poll a node from the PQ, it will mean that that specific node is going to be a part of our MST and therefore, we can add its cost to the finalCost variable

- Key points to remember: 
  - at each step is to first check if the node that we are processing, say the node we removed/polled from the PQ, if that has been visited or not. If it has been visited, we obviously ignore it and move onto the next node
  - we run a loop for each node's neighbors (starting from our chosen origin node) and add them to the PQ (as long as they haven't already been visited)


## Code
- here, I am referring to Apna College/Shraddha Khapra's code where she creates her own 'Edge' class. In real world scenarios, this piece of code might differ.  

```java
// TC: O(ElogE)
public static void primAlgo(ArrayList<Edge> graph[]) {
  boolean vis[] = new boolean[graph.length];
  PriorityQueue<Pair> pq = new PriorityQueue<>();
  pq.add(new Pair(0, 0));
  int cost = 0;

  while(!pq.isEmpty()) {
    Pair curr = pq.remove();
    if(!vis[curr.v]) {
      vis[curr.v] = true;
      cost += curr.wt;
      
      for(int i=0; i<graph[curr.v].size(); i++) {
        Edge e = graph[curr.v].get(i);
        if(!vis[e.dest]) {
          pq.add(new Pair(e.dest, e.wt));
        }
      }
    }
  }

  System.out.println(cost);
}
```