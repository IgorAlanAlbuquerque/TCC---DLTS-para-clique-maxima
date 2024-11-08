import os
import random
import numpy as np
import networkx as nx

def read_dimacs(filepath):
    """Lê um grafo no formato DIMACS e retorna um objeto NetworkX."""
    graph = nx.Graph()
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('p'):
                _, _, num_vertices, _ = line.split()
                num_vertices = int(num_vertices)
            elif line.startswith('e'):
                _, u, v = line.split()
                u = int(u) - 1  # Converter para índice baseado em 0
                v = int(v) - 1  # Converter para índice baseado em 0
                graph.add_edge(u, v)
    return graph


def read_clique(filepath):
    """Lê um arquivo com a clique e retorna uma lista de vértices."""
    with open(filepath, 'r') as file:
        for line in file:
            if '[' in line and ']' in line:
                # Faz o split usando 'm' e pega a parte após 'm'
                clique_part = line.split('m', 1)[1].strip()
                # Pega tudo antes do '[' para obter somente os números da clique
                clique_part = clique_part.split('[')[0].strip()
                clique = clique_part.split()  # Divide a string em uma lista de números
                clique = [int(node) - 1 for node in clique]  # Converte para índices baseados em 0
                return clique


def create_adjacency_matrix(graph):
    """Cria uma matriz de adjacência a partir de um grafo NetworkX."""
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    return np.array(adjacency_matrix)

def update_matrix(adjacency_matrix, vertex):
    """
    Atualiza a matriz de adjacência para um vértice específico.
    
    Parâmetros:
    adjacency_matrix (numpy.ndarray): Matriz de adjacência a ser atualizada.
    vertex (int): Vértice específico para a atualização.
    
    Retorna:
    numpy.ndarray: Matriz de adjacência atualizada.
    """
    num_vertices = adjacency_matrix.shape[0]
    
    # Obter os vizinhos do vértice
    neighbors = np.where(adjacency_matrix[vertex] == 1)[0]
    
    # Zerar linhas e colunas dos vértices que não são vizinhos do vértice
    for i in range(num_vertices):
        if i != vertex and i not in neighbors:
            adjacency_matrix[i, :] = 0
            adjacency_matrix[:, i] = 0
            
    # Zerar a linha e a coluna do vértice
    adjacency_matrix[vertex, :] = 0
    adjacency_matrix[:, vertex] = 0
            
    return adjacency_matrix

def save_matrix_to_file(matrix, filepath):
    """Salva a matriz de adjacência em um arquivo com o prefixo 'AA '."""
    with open(filepath, 'a') as file:
        for row in matrix:
            file.write("AA " + ", ".join(map(str, row)) + "\n")

def process_graph_and_clique(graph_filepath, clique, output_filepath):
    """Processa um grafo e sua clique, atualizando a matriz de adjacência e salvando o resultado."""
    # Ler o grafo e a clique dos arquivos
    G = read_dimacs(graph_filepath)
    
    # Criar a matriz de adjacência a partir do grafo
    adj_matrix = create_adjacency_matrix(G)
    
    # Salvar a matriz de adjacência original
    save_matrix_to_file(adj_matrix, output_filepath)
    
    current_clique = [0] * adj_matrix.shape[0]
    with open(output_filepath, 'a') as file:
        file.write("AA cliqueatual " + " ".join(map(str, current_clique)) + "\n")
    
    # Iterar sobre os vértices da clique e atualizar a matriz
    for vertex in clique:
        current_clique[vertex] = 1
        with open(output_filepath, 'a') as file:
            file.write(f"AA movimento {vertex + 1}\n")
        
        adj_matrix = update_matrix(adj_matrix, vertex)
        save_matrix_to_file(adj_matrix, output_filepath)
        
        with open(output_filepath, 'a') as file:
            file.write("AA cliqueatual " + " ".join(map(str, current_clique)) + "\n")


def p_process_graph_and_clique(graph_filepath, clique_filepath, output_filepath):
    """Cria múltiplas ramificações para a mesma clique e processa cada variação."""
    clique = read_clique(clique_filepath)
    
    # Processar a clique na ordem original
    process_graph_and_clique(graph_filepath, clique, output_filepath)
    

graphs_dir = 'graphs'
cliques_dir = 'cliques'
output_dir = 'train_graphs'

# Criar o diretório de saída se não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Processar todos os arquivos de grafo e clique correspondentes
for graph_filename in os.listdir(graphs_dir):
    if graph_filename.endswith('.dimacs'):
        graph_num = graph_filename.split('_')[1].split('.')[0]
        clique_filename = f"clique_graph_{graph_num}.dimacs"
        graph_filepath = os.path.join(graphs_dir, graph_filename)
        clique_filepath = os.path.join(cliques_dir, clique_filename)
        output_filepath = os.path.join(output_dir, f"instancia_{graph_filename}")
        
        if os.path.exists(clique_filepath):
            p_process_graph_and_clique(graph_filepath, clique_filepath, output_filepath)
        else:
            print(f"Clique file not found for graph: {graph_filename}")
