from line_profiler import profile
from tensorflow.keras.models import load_model
import time
import networkx as nx
import numpy as np

tam = 30
final_model_dnn_path = f"modelos/dnn_model_{tam}.keras"
final_model_value_path = f"modelos/dnn_value_model_{tam}.keras"
model_branching = load_model(final_model_dnn_path)
model_bounding = load_model(final_model_value_path)


def build_state_representation(Q, K, G_dict_of_lists, tam):
    vertex_list = list(K)
    len_k = len(vertex_list)

    vertex_to_idx = {vertex: i for i, vertex in enumerate(vertex_list)}

    adj_matrix = np.zeros((tam, tam), dtype=np.float32)
    clique_mask = np.zeros(tam, dtype=np.float32)

    if not Q:
        return adj_matrix, clique_mask

    v_q = Q[-1]

    G_adj_sets_for_K = {
        node_k: set(G_dict_of_lists.get(node_k, [])) for node_k in vertex_list
    }

    neighbors_of_vq_in_G_set = set(G_dict_of_lists.get(v_q, []))

    neighbors_of_vq_in_K = neighbors_of_vq_in_G_set & K

    if v_q in vertex_to_idx:
        idx_vq = vertex_to_idx[v_q]
        v_q_is_in_NvqK = v_q in neighbors_of_vq_in_K

        for k_idx in range(len_k):
            vk = vertex_list[k_idx]
            vk_is_in_NvqK = vk in neighbors_of_vq_in_K

            if v_q_is_in_NvqK or vk_is_in_NvqK:
                if vk in neighbors_of_vq_in_G_set:
                    adj_matrix[idx_vq, k_idx] = 1.0

            if vk_is_in_NvqK or v_q_is_in_NvqK:
                if v_q in G_adj_sets_for_K[vk]:
                    adj_matrix[k_idx, idx_vq] = 1.0

    Q_set = set(Q)
    for i in range(len_k):
        if vertex_list[i] in Q_set:
            clique_mask[i] = 1.0

    return adj_matrix, clique_mask


def build_inputs_for_model(adj_matrix, clique_mask):
    tam = adj_matrix.shape[0]

    if tam == 0:
        inputs = []
    else:
        inputs = np.vsplit(adj_matrix, tam)

    inputs.append(clique_mask[np.newaxis, :])

    return inputs


@profile
def dnn_bounding(Q, K, G, tam):
    adj_matrix, clique_mask = build_state_representation(Q, K, G, tam)
    inputs = build_inputs_for_model(adj_matrix, clique_mask)
    pred = model_bounding.predict(inputs, verbose=0)
    valor = pred[0, 0]  # extrai o escalar do array
    return int(round(valor))


@profile
def get_branching_dnn_predictions(Q, K, G_permuted, tam):
    if not K:  # Se K estiver vazio, não há o que pontuar.
        return np.array(
            []
        )  # Retorna um array vazio ou None, dependendo de como o chamador lida.

    adj_matrix, clique_mask = build_state_representation(Q, K, G_permuted, tam)
    inputs = build_inputs_for_model(adj_matrix, clique_mask)
    pred_raw = model_branching.predict(inputs, verbose=0)  # Saída é (1, tam)
    return pred_raw[0]


def should_expand(Q_new, K_new, C, G, tam):
    est_total = len(Q_new) + dnn_bounding(Q_new, K_new, G, tam)
    return est_total > len(C)


def neighbors(G, v):
    return set(G[v])


def pre_process_instance(G_original):
    original_node_labels = sorted(
        G_original.keys(),
        key=lambda v_orig: -len(G_original.get(v_orig, [])),
        reverse=False,
    )

    p_vertex_position = {
        v_orig: i_permuted for i_permuted, v_orig in enumerate(original_node_labels)
    }
    p_vertex_at_position = {
        i_permuted: v_orig for i_permuted, v_orig in enumerate(original_node_labels)
    }

    permuted_G = {}
    num_nodes = len(original_node_labels)
    for i_permuted in range(num_nodes):
        v_orig = p_vertex_at_position[i_permuted]
        original_neighbors = G_original.get(v_orig, [])
        permuted_G[i_permuted] = {
            p_vertex_position[u_orig] for u_orig in original_neighbors
        }

    C_permuted = []
    K_permuted_initial = set(permuted_G.keys())  # Nós de 0 a num_nodes-1
    S = [
        ([], K_permuted_initial)
    ]  # Q inicial vazio, K inicial com todos os nós permutados

    return C_permuted, S, permuted_G, p_vertex_at_position


def post_process_instance(C_permuted, p_vertex_at_position):
    return [p_vertex_at_position[i_permuted] for i_permuted in C_permuted]


def MCBB_DLTS_teste_tempo(
    G_original, tam
):  # A função que você quer analisar linha por linha

    start_time = time.time()  # Você ainda pode manter sua medição de tempo total
    nodes_expandidos_count = 0

    C_permuted, S, G_permuted, map_vertex_at_position = pre_process_instance(G_original)

    scores_branching_atuais = None

    while S:
        Q_raiz_perm, K_raiz_perm = S.pop()
        nodes_expandidos_count += 1

        Q_caminho_perm = list(Q_raiz_perm)
        K_caminho_perm = set(K_raiz_perm)

        # iteracoes_desde_ultimo_refresh_branching = 0 # Resetado no seu código original
        scores_branching_atuais = None  # Resetado no seu código original

        while K_caminho_perm:
            extensao_estimada_clique = dnn_bounding(
                Q_caminho_perm, K_caminho_perm, G_permuted, tam
            )

            if not (len(C_permuted) < len(Q_caminho_perm) + extensao_estimada_clique):
                break

            scores_branching_atuais = get_branching_dnn_predictions(
                Q_caminho_perm, K_caminho_perm, G_permuted, tam
            )

            v_perm = None
            # Sua lógica original para selecionar v_perm com base nos scores:
            if scores_branching_atuais is not None:
                scores_candidatos_K = {
                    idx_node: scores_branching_atuais[idx_node]
                    for idx_node in K_caminho_perm
                    # Garante que o índice do nó existe no array de scores e é válido
                    if 0 <= idx_node < len(scores_branching_atuais)
                }

                if scores_candidatos_K:
                    v_perm = max(scores_candidatos_K, key=scores_candidatos_K.get)

            if v_perm is None:
                break

            K_caminho_perm_sem_v = set(K_caminho_perm)  # Cria cópia
            K_caminho_perm_sem_v.remove(v_perm)

            if K_caminho_perm_sem_v:
                if should_expand(
                    Q_caminho_perm, K_caminho_perm_sem_v, C_permuted, G_permuted, tam
                ):
                    S.append((list(Q_caminho_perm), K_caminho_perm_sem_v))

            Q_caminho_perm.append(v_perm)
            K_caminho_perm = K_caminho_perm_sem_v.intersection(
                neighbors(G_permuted, v_perm)
            )

        if len(C_permuted) < len(Q_caminho_perm):
            C_permuted = list(Q_caminho_perm)

    elapsed_time = time.time() - start_time
    C_final_original = post_process_instance(C_permuted, map_vertex_at_position)

    return C_final_original, nodes_expandidos_count, elapsed_time


# Exemplo de como chamar sua função para que o profiler funcione:
if __name__ == "__main__":
    G = nx.gnp_random_graph(30, 0.9)
    G_dict = nx.to_dict_of_lists(G)

    print("Iniciando execução para profiling...")
    clique, nos_exp, tempo_total = MCBB_DLTS_teste_tempo(G_dict, 30)
    print(
        f"Execução concluída. Clique: {clique}, Nós: {nos_exp}, Tempo: {tempo_total:.4f}s"
    )
    # O resultado do line_profiler será impresso no console ao final pelo kernprof
