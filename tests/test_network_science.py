import networkx as nx
import pandas as pd
import pytest

from solutions.network_science import (
    barabasi_albert_summary,
    centrality_rankings,
    clustering_summary,
    connected_component_sizes,
    degree_statistics,
    detect_greedy_communities,
    erdos_renyi_summary,
    graph_from_edge_list,
    random_graph_phase_transition,
    robustness_comparison,
    shortest_path_lengths,
)


def test_graph_from_edge_list_builds_weighted_graph():
    edges = pd.DataFrame(
        {
            "source": ["A", "B", "A"],
            "target": ["B", "C", "C"],
            "weight": [1.5, 2.0, 3.0],
        }
    )

    graph = graph_from_edge_list(edges, weight_col="weight")

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3
    assert graph["A"]["B"]["weight"] == pytest.approx(1.5)


def test_graph_from_edge_list_rejects_missing_columns():
    edges = pd.DataFrame({"source": ["A"], "destination": ["B"]})

    with pytest.raises(ValueError):
        graph_from_edge_list(edges)


def test_degree_statistics_for_path_graph():
    graph = nx.path_graph(4)

    result = degree_statistics(graph)

    assert result["n_nodes"] == 4
    assert result["n_edges"] == 3
    assert result["density"] == pytest.approx(0.5)
    assert result["average_degree"] == pytest.approx(1.5)
    assert result["degree_distribution"] == {1: 2, 2: 2}


def test_connected_component_sizes():
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (3, 4)])
    graph.add_node(5)

    assert connected_component_sizes(graph) == [2, 2, 1]


def test_shortest_path_lengths():
    graph = nx.path_graph(4)

    assert shortest_path_lengths(graph, source=0) == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }


def test_shortest_path_lengths_rejects_missing_source():
    graph = nx.path_graph(4)

    with pytest.raises(ValueError):
        shortest_path_lengths(graph, source=99)


def test_clustering_summary_for_triangle():
    graph = nx.complete_graph(3)

    result = clustering_summary(graph)

    assert result["average_clustering"] == pytest.approx(1.0)
    assert result["local_clustering"] == {0: 1.0, 1: 1.0, 2: 1.0}


def test_erdos_renyi_summary_for_complete_probability():
    result = erdos_renyi_summary(
        n_nodes=5,
        probability=1.0,
        seed=2026,
    )

    assert result["n_nodes"] == 5
    assert result["n_edges"] == 10
    assert result["expected_average_degree"] == pytest.approx(4.0)
    assert result["observed_average_degree"] == pytest.approx(4.0)


def test_random_graph_phase_transition_endpoints():
    result = random_graph_phase_transition(
        n_nodes=10,
        probabilities=[0.0, 1.0],
        seed=2026,
    )

    assert result["largest_component_size"].tolist() == [1, 10]
    assert result["largest_component_fraction"].tolist() == [0.1, 1.0]
    assert result["n_edges"].tolist() == [0, 45]


def test_barabasi_albert_summary():
    result = barabasi_albert_summary(
        n_nodes=10,
        edges_per_new_node=2,
        seed=2026,
    )

    assert result["n_nodes"] == 10
    assert result["n_edges"] == 16
    assert sum(result["degree_distribution"].values()) == 10
    assert result["average_degree"] == pytest.approx(3.2)


def test_centrality_rankings_for_star_graph():
    graph = nx.star_graph(4)

    rankings = centrality_rankings(graph, top_n=1)

    assert set(rankings["metric"]) == {
        "degree",
        "betweenness",
        "closeness",
        "eigenvector",
    }
    assert rankings["node"].tolist() == [0, 0, 0, 0]


def test_robustness_comparison_targeted_attack_on_star_graph():
    graph = nx.star_graph(4)

    result = robustness_comparison(
        graph,
        fractions=[0.0, 0.2],
        seed=2026,
    )

    targeted_row = result[
        (result["fraction_removed"] == 0.2)
        & (result["strategy"] == "targeted_attack")
    ].iloc[0]

    assert targeted_row["removed_nodes"] == 1
    assert targeted_row["largest_component_size"] == 1
    assert targeted_row["largest_component_fraction"] == pytest.approx(0.2)


def test_detect_greedy_communities_for_two_triangles():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 3),
        ]
    )

    result = detect_greedy_communities(graph)
    labels = result["labels"]

    assert result["n_communities"] == 2
    assert result["modularity"] > 0
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
