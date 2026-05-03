"""Reference solutions for network science graph analytics problems."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def graph_from_edge_list(
    edges: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str | None = None,
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
    """Build a NetworkX graph from an edge-list DataFrame."""
    required_columns = {source_col, target_col}

    if weight_col is not None:
        required_columns.add(weight_col)

    missing_columns = required_columns - set(edges.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"edges is missing required columns: {missing}")

    graph: nx.Graph | nx.DiGraph

    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    for _, row in edges.iterrows():
        source = row[source_col]
        target = row[target_col]

        if weight_col is None:
            graph.add_edge(source, target)
        else:
            graph.add_edge(source, target, weight=float(row[weight_col]))

    return graph


def degree_statistics(graph: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    """Return basic degree and density statistics for a graph."""
    degrees = [degree for _, degree in graph.degree()]
    degree_distribution = dict(sorted(Counter(degrees).items()))

    if degrees:
        average_degree = float(np.mean(degrees))
    else:
        average_degree = 0.0

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "average_degree": average_degree,
        "degree_distribution": degree_distribution,
    }


def _component_sets(graph: nx.Graph | nx.DiGraph) -> list[set[Any]]:
    """Return connected components, using weak components for directed graphs."""
    if graph.is_directed():
        return [set(component) for component in nx.weakly_connected_components(graph)]

    return [set(component) for component in nx.connected_components(graph)]


def connected_component_sizes(graph: nx.Graph | nx.DiGraph) -> list[int]:
    """Return connected-component sizes in descending order."""
    sizes = [len(component) for component in _component_sets(graph)]

    return sorted(sizes, reverse=True)


def shortest_path_lengths(
    graph: nx.Graph | nx.DiGraph,
    source: Any,
) -> dict[Any, int]:
    """Return shortest-path lengths from a source node."""
    if source not in graph:
        raise ValueError("source must be a node in graph.")

    return dict(nx.single_source_shortest_path_length(graph, source))


def clustering_summary(graph: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    """Return local and average clustering coefficients."""
    work_graph = graph.to_undirected() if graph.is_directed() else graph

    if work_graph.number_of_nodes() == 0:
        return {"local_clustering": {}, "average_clustering": 0.0}

    local_clustering = nx.clustering(work_graph)

    return {
        "local_clustering": local_clustering,
        "average_clustering": nx.average_clustering(work_graph),
    }


def erdos_renyi_summary(
    n_nodes: int,
    probability: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate an Erdos-Renyi graph and summarize its average degree."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")

    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1.")

    graph = nx.gnp_random_graph(n_nodes, probability, seed=seed)
    degrees = [degree for _, degree in graph.degree()]

    return {
        "graph": graph,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "expected_average_degree": probability * (n_nodes - 1),
        "observed_average_degree": float(np.mean(degrees)),
    }


def random_graph_phase_transition(
    n_nodes: int,
    probabilities: Iterable[float],
    seed: int | None = None,
) -> pd.DataFrame:
    """Track largest-component size across Erdos-Renyi probabilities."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive.")

    rows = []

    for index, probability in enumerate(probabilities):
        if not 0 <= probability <= 1:
            raise ValueError("All probabilities must be between 0 and 1.")

        graph_seed = None if seed is None else seed + index
        graph = nx.gnp_random_graph(n_nodes, probability, seed=graph_seed)
        component_sizes = connected_component_sizes(graph)
        largest_component_size = max(component_sizes, default=0)

        rows.append(
            {
                "probability": probability,
                "n_edges": graph.number_of_edges(),
                "largest_component_size": largest_component_size,
                "largest_component_fraction": largest_component_size / n_nodes,
            }
        )

    return pd.DataFrame(rows)


def barabasi_albert_summary(
    n_nodes: int,
    edges_per_new_node: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a Barabasi-Albert graph and summarize its degree distribution."""
    if not 1 <= edges_per_new_node < n_nodes:
        raise ValueError("edges_per_new_node must satisfy 1 <= m < n_nodes.")

    graph = nx.barabasi_albert_graph(
        n=n_nodes,
        m=edges_per_new_node,
        seed=seed,
    )
    degrees = [degree for _, degree in graph.degree()]

    return {
        "graph": graph,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "degree_distribution": dict(sorted(Counter(degrees).items())),
        "max_degree": max(degrees),
        "average_degree": float(np.mean(degrees)),
    }


def centrality_rankings(
    graph: nx.Graph | nx.DiGraph,
    top_n: int = 5,
) -> pd.DataFrame:
    """Rank nodes by degree, betweenness, closeness, and eigenvector centrality."""
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node.")

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    centralities = {
        "degree": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(graph, normalized=True),
        "closeness": nx.closeness_centrality(graph),
        "eigenvector": nx.eigenvector_centrality(graph, max_iter=1_000),
    }

    rows = []

    for metric, scores in centralities.items():
        ranked_nodes = sorted(
            scores.items(),
            key=lambda item: (-item[1], str(item[0])),
        )

        for rank, (node, score) in enumerate(ranked_nodes[:top_n], start=1):
            rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "node": node,
                    "score": float(score),
                }
            )

    return pd.DataFrame(rows)


def _largest_component_size_after_removal(
    graph: nx.Graph | nx.DiGraph,
    nodes_to_remove: Sequence[Any],
) -> int:
    """Remove nodes and return the resulting largest-component size."""
    work_graph = graph.copy()
    work_graph.remove_nodes_from(nodes_to_remove)

    return max(connected_component_sizes(work_graph), default=0)


def robustness_comparison(
    graph: nx.Graph | nx.DiGraph,
    fractions: Iterable[float],
    seed: int | None = None,
) -> pd.DataFrame:
    """Compare random node failure with targeted high-degree attack."""
    n_nodes = graph.number_of_nodes()

    if n_nodes == 0:
        raise ValueError("graph must contain at least one node.")

    nodes = list(graph.nodes())
    rng = np.random.default_rng(seed)
    targeted_order = [
        node
        for node, _ in sorted(
            graph.degree(),
            key=lambda item: (-item[1], str(item[0])),
        )
    ]

    rows = []

    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError("All fractions must be between 0 and 1.")

        remove_count = int(np.floor(fraction * n_nodes))

        if remove_count == 0:
            random_nodes: list[Any] = []
        else:
            random_nodes = list(
                rng.choice(nodes, size=remove_count, replace=False),
            )

        targeted_nodes = targeted_order[:remove_count]

        for strategy, nodes_to_remove in (
            ("random_failure", random_nodes),
            ("targeted_attack", targeted_nodes),
        ):
            largest_component_size = _largest_component_size_after_removal(
                graph,
                nodes_to_remove,
            )

            rows.append(
                {
                    "fraction_removed": fraction,
                    "strategy": strategy,
                    "removed_nodes": remove_count,
                    "largest_component_size": largest_component_size,
                    "largest_component_fraction": largest_component_size / n_nodes,
                }
            )

    return pd.DataFrame(rows)


def detect_greedy_communities(graph: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    """Detect communities using greedy modularity maximization."""
    work_graph = graph.to_undirected() if graph.is_directed() else graph

    if work_graph.number_of_nodes() == 0:
        return {"labels": {}, "modularity": 0.0, "n_communities": 0}

    communities = list(nx.community.greedy_modularity_communities(work_graph))
    labels = {}

    for community_id, community in enumerate(communities):
        for node in community:
            labels[node] = community_id

    return {
        "labels": labels,
        "modularity": float(nx.community.modularity(work_graph, communities)),
        "n_communities": len(communities),
    }
