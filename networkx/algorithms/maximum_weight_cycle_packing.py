import doctest
import concurrent.futures
import logging
import networkx as nx
import numpy as np
from networkx.algorithms import cycles
LOG_FORMAT= "%(levelname)s, time: %(asctime)s , line: %(lineno)d - %(message)s"
logging.basicConfig(filename="logging.log",level=logging.DEBUG,format=LOG_FORMAT)
logger = logging.getLogger()
import time

"""REUT HADAD & TAL SOMECH"""

"""
This is an implementation for two different algorithms described on "MAXIMUM WEIGHT CYCLE PACKING IN DIRECTED GRAPHS,
WITH APPLICATION TO KIDNEY EXCHANGE PROGRAMS" article.
The article points on two algorithms that solves kidney exchange problems, which can be modelled as cycle packing
problems in a directed graph, involving cycles of length 2, 3, or even longer.
In the article we focus on the maximal exchange of circles of size 2 and 3 vertices, we demonstrate an approximation
algorithm and an exact algorithm for this problem.
"""
"""article title: MAXIMUM WEIGHT CYCLE PACKING IN DIRECTED GRAPHS,WITH APPLICATION TO KIDNEY EXCHANGE PROGRAMS
authors:Biro, P. and Manlove, D.F. and Rizzi, R.
year:(2009)
link:http://eprints.gla.ac.uk/25732/
"""


def maximum_weight_cycle_packing_improved(graph: nx.DiGraph, k: int,imp) -> list:
    """
    Algorithm - the algorithm finds the exact maximum weight k-way exchanges using reduction from directed graph to non directed
    graph
    "Algorithm 2 - Exact algorithm for kidney exchange programs" by Biro, P. and Manlove, D.F. and Rizzi, R.
    Returns the list of max weighted exchanges of directed weighted graph 'G'

    A directed weighted graph is a graph in which every edge is one sided and weighted
    for example an edge from node 1->2 with a weight of 5,an k-way exchange
    is a circle within a graph containing at most k nodes.
    max weighted exchange is a circle with the most weighted edges from every node in the circle

    Parameters
    -----------
    graph- NetworkX DiGraph
        Directed graph with weights

    Returns
    -----------
    Lst: list of lists
        Each list in lst contaning the nodes which make up the circle with the highest weights sum
    Examples
    -----------


    >>> Digraph=nx.DiGraph()
    >>> Digraph.add_nodes_from([1,2,3,5,6,7,8])
    >>> Digraph.add_weighted_edges_from([(1,8,2),(8,1,4),(2,1,5),(1,3,4),(3,8,2),(8,2,3),(8,5,4),(5,7,3),(7,6,2),(6,5,4)])
    >>> print(len(maximum_weight_cycle_packing_improved(Digraph,3,"Process"))) #[1,8,2] [6,5,7] [1,3,8] , can be only 2 but in any order
    2
    >>> Digraph =nx.DiGraph()
    >>> Digraph.add_nodes_from([1,2,3,4])
    >>> Digraph.add_weighted_edges_from([(2,1,3),(1,3,1),(3,2,2),(3,4,5),(4,3,9)])
    >>> print(len(maximum_weight_cycle_packing_improved(Digraph,2,"Process")))#[3,4] or [4,3]
    1
    >>> graphEX3 = nx.DiGraph()
    >>> graphEX3.add_nodes_from([10,11,12,13,14,15,16])
    >>> Digraph.add_weighted_edges_from([(10,11,10),(11,12,5),(12,13,6),(13,10,4),(11,14,2),(14,16,3),(16,15,8),(15,14,6)])
    >>> print(maximum_weight_cycle_packing_improved(graphEX3, 3"Thread"))
    []

    Notes
    -----------
    Algorithm - the algorithm finds maximum weight k-way exchanges using reduction from directed graph to not directed graph by
    the algorithm in the published article Exact-complete algorithm for kidney exchange programs"
    Refrences
    ----------
    Algorithm 1 - 'MAXIMUM WEIGHT CYCLE PACKING IN DIRECTED GRAPHS, WITH APPLICATION TO KIDNEY EXCHANGE PROGRAMS' by Biro, P. and Manlove, D.F. and Rizzi, R. http://eprints.gla.ac.uk/25732/
    """

    Ys, cycles = create_Ys(graph, k)

    max_cycles = []
    max_weight = 0
    if imp == "Threads":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # for Y in Ys:
            Ys_results = [executor.submit(get_exchanges, Y, graph) for Y in Ys]
            for res in concurrent.futures.as_completed(Ys_results):
                exchange_res = res.result()
                weight = exchange_res[0]
                if weight > max_weight:
                    max_cycles = exchange_res[1]
                    max_weight = weight
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # for Y in Ys:
            Ys_results = [executor.submit(get_exchanges, Y, graph) for Y in Ys]
            for res in concurrent.futures.as_completed(Ys_results):
                exchange_res = res.result()
                weight = exchange_res[0]
                if weight > max_weight:
                    max_cycles = exchange_res[1]
                    max_weight = weight
    return max_cycles



def get_exchanges(Y, graph):
    logger.debug("Starting exchanges, Y: "+str(Y))
    X = []
    seen_Y = set()
    ans_graph = nx.Graph()
    #   creating the nodes in the graph graph
    #   adding the nodes in the graph
    for edge in Y:
        if (edge[1], edge[0]) not in ans_graph.nodes:  # If we have (8,1) and (1,8) in Y we only want 1
            ans_graph.add_node((edge[0], edge[1]))
            seen_Y.add(edge[0])
            seen_Y.add(edge[1])
            if (edge[0], edge[1]) in graph.edges and (edge[1], edge[0]) in graph.edges:
                weight = graph.get_edge_data(edge[0], edge[1])["weight"] + graph.get_edge_data(edge[1], edge[0])[
                    "weight"]
                ans_graph.add_edge((edge[0], edge[1]), (edge[0], edge[1]), weight=weight, cycle=[edge[0], edge[1]])
    for edge in graph.edges:
        if edge[0] not in seen_Y and edge[0] not in X:
            X.append((edge[0]))
            ans_graph.add_node((edge[0]))
    with concurrent.futures.ThreadPoolExecutor() as exect:
        exect.submit(connect_2cycles, X, graph, ans_graph)
        exect.submit(connect_3cycles, X, Y, graph, ans_graph)
    exchanges = list(nx.max_weight_matching(ans_graph))
    if len(exchanges) == 0 and ans_graph.number_of_edges() == 1:  # for the use-case of only self connected edge
        exchanges = [list(ans_graph.edges)[0]]
    final_cycles = []
    temp_max = 0
    logger.info("finishing exchanges, Y: "+Y)
    for cyc in exchanges:
        ed = ans_graph.get_edge_data(cyc[0], cyc[1])
        temp_max = temp_max + ed["weight"]
        final_cycles.append(ed["cycle"])
    #Log.debug("finishing exchanges, Y: "+Y)
    return temp_max, final_cycles


def connect_2cycles(X, graph, ans_graph):
    for i in range(len(X)):  # creating the edges in the graph by going through the 2-circles
        for j in range(i + 1, len(X)):
            if (X[i], X[j]) in graph.edges and (X[j], X[i]) in graph.edges:
                weight = graph.get_edge_data(X[i], X[j])["weight"] + graph.get_edge_data(X[j], X[i])["weight"]
                ans_graph.add_edge((X[i]), (X[j]), weight=weight, cycle=[X[i], X[j]])


def connect_3cycles(X, Y, graph, ans_graph):
    #   creating the edges in the graph by going through the 3-circles
    for k in range(len(X)):
        for nodes in ans_graph.nodes:
            if isinstance(nodes, int):
                continue
            else:
                i = nodes[0]
                j = nodes[1]
                weight = 0
                if (X[k], i) in graph.edges and (j, X[k]) in graph.edges:
                    weight = graph.get_edge_data(X[k], i)["weight"] + graph.get_edge_data(j, X[k])["weight"] + \
                             graph.get_edge_data(i, j)["weight"]
                elif (X[k], j) in graph.edges and (i, X[k]) in graph.edges:
                    weight = graph.get_edge_data(X[k], j)["weight"] + graph.get_edge_data(i, X[k])["weight"] + \
                             graph.get_edge_data(j, i)["weight"]
                if weight > 0:
                    ans_graph.add_edge((X[k]), (i, j), weight=weight, cycle=[j, i, X[k]])


def simple_cycles(G, limit):
    """
    >>> Digraph=nx.DiGraph()
    >>> Digraph.add_nodes_from([1,2,3,5,6,7,8])
    >>> Digraph.add_weighted_edges_from([(1,8,2),(8,1,4),(2,1,5),(1,3,4),(3,8,2),(8,2,3),(8,5,4),(5,7,3),(7,6,2),(6,5,4)])
    >>> Ys=simple_cycles(Digraph,3)
    >>> a=[y for y in Ys]
    >>> print(len(a)) #- the known product is supposed to be composed of 27 permutation
    4
    """
    subG = type(G)(G.edges())
    sccs = list(nx.strongly_connected_components(subG))
    while sccs:
        scc = sccs.pop()
        startnode = scc.pop()
        path = [startnode]
        blocked = set()
        blocked.add(startnode)
        stack = [(startnode, list(subG[startnode]))]

        while stack:
            thisnode, nbrs = stack[-1]

            if nbrs and len(path) <= limit:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(subG[nextnode])))
                    blocked.add(nextnode)
                    continue
            if not nbrs or len(path) >= limit:
                blocked.remove(thisnode)
                stack.pop()
                path.pop()
        subG.remove_node(startnode)
        H = subG.subgraph(scc)
        sccs.extend(list(nx.strongly_connected_components(H)))


def create_Ys(graph, k):
    """This function is used to create the cartesian product of the 3-cycles
    >>> Digraph=nx.DiGraph()
    >>> Digraph.add_nodes_from([1,2,3,5,6,7,8])
    >>> Digraph.add_weighted_edges_from([(1,8,2),(8,1,4),(2,1,5),(1,3,4),(3,8,2),(8,2,3),(8,5,4),(5,7,3),(7,6,2),(6,5,4)])
    >>> Ys,_=create_Ys(Digraph,3)
    >>> print(len(Ys)) #- the known product is supposed to be composed of 27 permutation
    27
    >>> Digraph =nx.DiGraph()
    >>> Digraph.add_nodes_from([1,2,3,4])
    >>> Digraph.add_weighted_edges_from([(2,1,3),(1,3,1),(3,2,2),(3,4,5),(4,3,9)])
    >>> print(len(create_Ys(Digraph,3))) #- the known product is supposed to be composed of 1 permutation
    2
    """
    temp_cycles = simple_cycles(graph, k)  # nx.recursive_simple_cycles(graph)
    cycles = []
    for cycle in temp_cycles:
        if len(cycle) == k:
            cycles.append(cycle)
    perm_arr = np.ndarray(shape=(len(cycles), k), dtype=list)
    for cyc_idx in range(len(cycles)):
        cyc = cycles[cyc_idx]
        for ed_idx in range(len(cyc)):
            mid = (cyc[ed_idx], cyc[(ed_idx + 1) % len(cyc)])
            perm_arr[cyc_idx][ed_idx] = mid
    mesh = []
    if len(perm_arr) > 0:
        mesh = np.array(np.meshgrid(*perm_arr))
        mesh = mesh.T.reshape(-1, len(mesh))

    return mesh, cycles


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # itertools.ne

    doctest.testmod()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
