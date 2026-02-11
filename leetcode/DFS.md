
# DFS — What It Really Solves

**DFS answers one fundamental question: "What can I reach from here?"**

Everything else is a variation of that idea.

---

## The 5 Things DFS Does

### 1. Can A reach B? → Reachability & Components
"Are these connected? How many groups exist?"
Grids, graphs, flood fill, islands — all the same thing.

### 2. Is there a loop? → Cycle Detection
"Can I reach where I already am?"
If you revisit a node that's still in progress — cycle.

### 3. What order should I do things? → Topological Sort
"If A must come before B, what's a valid sequence?"
Finish exploring a node → record it → reverse at the end.

### 4. Can I find a valid assignment? → Backtracking
"Try a choice, go deep, undo if stuck."
Permutations, N-Queens, Sudoku, word search — all this pattern.

### 5. What's the answer for this subtree? → Tree / DP
"Compute children first, then combine."
Heights, diameters, longest paths, subtree sums.

---

## When NOT to Use DFS

**Shortest path** → BFS.
**Level-by-level** → BFS.
Pretty much everything else → DFS is fine.

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


