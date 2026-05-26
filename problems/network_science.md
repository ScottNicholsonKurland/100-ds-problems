# Network Science Graph Analytics Problems

Implement each function in `solutions/network_science.py`. Use `networkx`, `numpy`, and `pandas` where appropriate. Keep random graph functions deterministic when a `seed` is supplied. Do not use network calls.

## 1. `graph_from_edge_list`

Write a function that builds a NetworkX graph from a pandas edge-list DataFrame.

Requirements:

- Accept source and target column names.
- Optionally accept a weight column and store weights as floats.
- Support directed and undirected graphs.
- Raise `ValueError` when required columns are missing.

## 2. `degree_statistics`

Write a function that summarizes graph degree structure.

Return:

- Number of nodes.
- Number of edges.
- Graph density.
- Average degree.
- Degree distribution as a dictionary mapping degree to node count.

## 3. `connected_component_sizes`

Write a function that returns connected-component sizes in descending order.

Requirements:

- Use connected components for undirected graphs.
- Use weakly connected components for directed graphs.
- Include isolated nodes as components of size 1.

## 4. `shortest_path_lengths`

Write a function that returns shortest-path lengths from a source node.

Requirements:

- Return a dictionary mapping reachable nodes to shortest-path distance.
- Raise `ValueError` if the source node is not in the graph.

## 5. `clustering_summary`

Write a function that summarizes clustering coefficients.

Return:

- Local clustering coefficient for each node.
- Average clustering coefficient for the graph.

For directed graphs, compute clustering on the undirected version.

## 6. `erdos_renyi_summary`

Write a function that generates an Erdos-Renyi random graph and summarizes it.

Inputs:

- `n_nodes`
- Edge probability
- Optional random seed

Return:

- Generated graph.
- Number of nodes.
- Number of edges.
- Expected average degree.
- Observed average degree.

Validate that node count is positive and probability is between 0 and 1.

## 7. `random_graph_phase_transition`

Write a function that evaluates the phase transition behavior of Erdos-Renyi graphs across multiple probabilities.

Return a pandas DataFrame with one row per probability containing:

- Probability.
- Number of edges.
- Largest component size.
- Largest component fraction.

Use deterministic seeds when a seed is supplied.

## 8. `barabasi_albert_summary`

Write a function that generates a Barabasi-Albert preferential attachment graph and summarizes it.

Inputs:

- `n_nodes`
- Number of edges each new node attaches with
- Optional random seed

Return:

- Generated graph.
- Number of nodes.
- Number of edges.
- Degree distribution.
- Maximum degree.
- Average degree.

Validate that `edges_per_new_node` satisfies `1 <= m < n_nodes`.

## 9. `centrality_rankings`

Write a function that ranks nodes by multiple centrality metrics.

Metrics:

- Degree centrality.
- Betweenness centrality.
- Closeness centrality.
- Eigenvector centrality.

Return a pandas DataFrame with metric name, rank, node, and score. Sort ties deterministically by node label.

## 10. `robustness_comparison`

Write a function that compares graph robustness under random failure and targeted attack.

For each removal fraction:

- Randomly remove nodes using the supplied seed.
- Remove highest-degree nodes for targeted attack.
- Report removed-node count.
- Report largest remaining component size.
- Report largest remaining component fraction.

Return results as a pandas DataFrame.

## 11. `detect_greedy_communities`

Write a function that detects communities using greedy modularity maximization.

Return:

- A dictionary mapping each node to a community label.
- Modularity score.
- Number of communities.

For directed graphs, run detection on the undirected version.
