import networkx as nx
import random
import os


def generate_random_graph(num_node_min, num_node_max, prob_min, prob_max):
    num_nodes = random.randint(num_node_min, num_node_max)
    edge_prob = random.uniform(prob_min, prob_max)

    while True:
        graph = nx.gnp_random_graph(num_nodes, edge_prob)
        if nx.is_connected(graph):
            return graph  # Retorna o grafo se for totalmente conexo

        # Se não for conexo, forçamos a conexão entre os componentes
        components = list(nx.connected_components(graph))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            graph.add_edge(u, v)  # Conectar componentes desconexos


def write_dimacs(graph, filepath):
    with open(filepath, 'w') as f:
        num_vertices = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        # Escrever o cabeçalho
        f.write(f"p edge {num_vertices} {num_edges}\n")

        # Escrever as arestas
        for u, v in graph.edges():
            f.write(f"e {u+1} {v+1}\n")


def generate_and_save_graphs_in_batches(num_graph, num_node_min, num_node_max,
                                        prob_min, prob_max, batch_size,
                                        directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for batch_start in range(0, num_graph, batch_size):
        batch_end = min(batch_start + batch_size, num_graph)
        for i in range(batch_start, batch_end):
            graph = generate_random_graph(num_node_min, num_node_max, prob_min,
                                          prob_max)
            filepath = os.path.join(directory, f"graph_{i + 1}.dimacs")
            write_dimacs(graph, filepath)
        print(f"{batch_end} grafos gerados e salvos...")


# Parâmetros para gerar os grafos
num_graph = 25000
num_node_min = 15
num_node_max = 30
prob_min = 0.3
prob_max = 0.8
batch_size = 100  # Ajuste o tamanho do lote conforme necessário

# Gerar e salvar grafos em lotes
print("Gerando e salvando grafos em lotes...")
generate_and_save_graphs_in_batches(num_graph, num_node_min, num_node_max,
                                    prob_min, prob_max, batch_size, "graphs")

print("Processo concluído.")
