
```python
# ----------------------------
# DFS Template (Recursive)
# ----------------------------
def dfs_recursive(start, graph):
    visited = set()

    def dfs(u):
        visited.add(u)
        # process u here (preorder)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
        # process u here (postorder)

    dfs(start)
    return visited
# ----------------------------
# DFS Template (Iterative Stack)
# ----------------------------
def dfs_stack(start, graph):
    visited = set()
    stack = [start]

    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        # process u here

        # push neighbors (reverse if you care about matching recursive order)
        for v in graph[u]:
            if v not in visited:
                stack.append(v)

    return visited
```

| # | LeetCode | Topic |
|---:|---|---|
| 1 | **200. Number of Islands** ⭐ **(Sequence Step 1)** | Grid DFS / connected components |
| 2 | 733. Flood Fill | Flood fill |
| 3 | 695. Max Area of Island | Component size / area |
| 4 | 130. Surrounded Regions | Boundary-connected regions |
| 5 | 417. Pacific Atlantic Water Flow | Multi-source DFS from borders |
| 6 | **207. Course Schedule** ⭐ **(Sequence Step 2)** | Directed cycle detection (colors) |
| 7 | **210. Course Schedule II** ⭐ **(Sequence Step 3)** | Topological ordering |
| 8 | 802. Find Eventual Safe States | Cycle detection variant (safe nodes) |
| 9 | 2360. Longest Cycle in a Graph | Cycle length with timestamps |
|10 | 684. Redundant Connection | Undirected cycle detection |
|11 | **785. Is Graph Bipartite?** ⭐ **(Sequence Step 4)** | 2-coloring / bipartite |
|12 | 886. Possible Bipartition | Bipartite constraints |
|13 | **1192. Critical Connections in a Network** ⭐ **(Sequence Step 5)** | Bridges (Tarjan low-link) |
|14 | 1568. Minimum Number of Days to Disconnect Island | “Disconnect” logic + DFS |
|15 | 332. Reconstruct Itinerary | Postorder build (Hierholzer-ish) |
|16 | **329. Longest Increasing Path in a Matrix** ⭐ **(Sequence Step 6)** | DFS + memoization (DP on DAG) |


