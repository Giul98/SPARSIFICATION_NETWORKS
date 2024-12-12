import networkx as nx
import numpy as np
from itertools import combinations, product
import matplotlib.pyplot as plt
import time

# Creo una tabella DP [v][b] per ogni nodo v con b che varia da 0 a k (numero massimo di collegamenti selezionabili)
# DP[v][b] contiene il massimo valore di log-verosimiglianza ottenibile scegliendo b collegamenti in ingresso al nodo v

def initialize_dp(graph, k):
    DP = {}
    for node in graph.nodes:
        DP[node] = [None] * (k + 1)  # Fino a k collegamenti per nodo + la possibilità di selezionare 0 collegamenti per quel nodo
    return DP


# Calcolo iterativo delle probabilità p(u, v)
def compute_probabilities(graph, log, max_iterations=100, tol=1e-5):
    """
    Calcola iterativamente le probabilità p(u, v) per ogni arco nel grafo.
    """
    p = {edge: 0.5 for edge in graph.edges}  # Inizializza p(u, v)
    p_prev = p.copy()          #Fa una copia p_prev delle probabilità iniziali per confrontare le probabilità attuali con quelle calcolate nell'iterazione precedente.

    for it in range(max_iterations):
        p_updates = {}      #dizionario p_updates per memorizzare le probabilità aggiornate di questa iterazione. (la chiave è l'arco (u,v) e il valore è la probabilità p(u,v))
        for u, v in graph.edges:
            # Calcola le liste di azioni A+ e A-
            A_plus = [action for action in log if v == action[0] and u in action[2]]   #cerca nel log delle azioni tutte le istanze in cui il nodo v è influenzato e u è uno degli influenzatori.
            A_minus = [action for action in log if v == action[0] and u not in action[2]]  #cerca nel log delle azioni tutte le istanze in cui v è influenzato, ma u non è uno degli influenzatori.

            # Somma le probabilità di successo
            # Per ogni azione in A+
            # Calcola la probabilità che v sia stato influenzato escludendo u (cioè considerando tutti gli altri influenzatori w).
            P_plus_sum = sum(
                1 - np.prod([(1 - p.get((w, v), 0)) for w in action[2] if w != u])    #La probabilità complementare (1−) indica che almeno uno degli altri influenzatori ha avuto successo.
                for action in A_plus
            )

            # Prodotto delle probabilità di fallimento
            # Per ogni azione in A-
            # Calcola la probabilità che v non sia stato influenzato considerando tutti gli influenzatori w tranne u.
            P_minus_prod = np.prod(
                np.prod([(1 - p.get((w, v), 0)) for w in action[2] if w != u])   #Moltiplico tutte queste probabilità insieme per ottenere P− , la probabilità complessiva che u non abbia influenzato v.
                for action in A_minus
            )

            denom = len(A_plus) + len(A_minus)   #Calcolo il denominatore come il numero totale di azioni in A+ e A−
            if denom > 0:
                p_updates[(u, v)] = P_plus_sum / denom  #Se ci sono azioni osservate (denom>0), aggiorna la probabilità p(u,v) come la media di P+.

        # Calcola il massimo cambiamento delle probabilità
        max_diff = max(abs(p_updates[edge] - p_prev[edge]) for edge in p_updates)   #Calcola la differenza massima tra le probabilità aggiornate p(u,v) e quelle dell'iterazione precedente
        p_prev.update(p_updates)  #aggiorna i valori nel dizionario

        # Convergenza raggiunta
        if max_diff < tol:
            break

    return p_prev


def compute_lambda(graph, node, log, b, p):
    """
    per ogni sotto insieme di collegamenti in ingresso al nodo di dimensione b (Dvb) calcolo la funzione di Log-Similarity del nodo
    lamda(Dvb) = somma della probabilità che almeno uno dei nodi in Dvb abbia influenzato v + probabilità che tutti i nodi
    NON in Dvb non abbiano influenzato v; questo per ogni nodo v
    """
    #estratti tutti gli archi entranti nel nodo v, cioè tutti gli archi che
    #possono potenzialmente essere considerati per la selezione del sottoinsieme Dv,b.
    in_edges = list(graph.in_edges(node, data=True))
    if b > len(in_edges):
        return -float('inf')  # Non è possibile selezionare più archi di quelli disponibili

    best_likelihood = -float('inf')  #inizializzazione

    for subset in combinations(in_edges, b):   #generati tutti i possibili sottoinsiemi di dimensione b degli archi entranti al nodo v.
        likelihood = 0
        for action in log:
            target, time, influencers = action
            if target != node:   #Solo le azioni che influenzano il nodo v vengono considerate.
                continue

            # Calcola la probabilità di successo e fallimento di influenzare un nodo
            prob_success = 1 - np.prod([(1 - p.get((u, target), 0)) for u in influencers if u in {u for u, _, _ in subset}])
            prob_success = max(prob_success, 1e-7)  # Per evitare problemi numerici

            #che nessun nodo non presente nel sottoinsieme abbia influenzato v.
            prob_failure = np.prod([(1 - p.get((u, target), 0)) for u in influencers if u not in {u for u, _, _ in subset}])
            prob_failure = max(prob_failure, 1e-7)  # Per evitare log(0)

            # Aggiungi al likelihood
            likelihood += np.log(prob_success) + np.log(prob_failure)

        # Confronta con la miglior likelihood trovata finora
        best_likelihood = max(best_likelihood, likelihood)

    return best_likelihood



# La funzione fill_dp riempie una tabella DP per ciascun nodo del grafo, dove ogni entry DP[node][b]
# rappresenta il valore della funzione λ calcolato per quel nodo e un sottoinsieme di archi entranti di dimensione b.

def fill_dp(graph, log, k, p):  # Aggiunto parametro p
    DP = initialize_dp(graph, k)
    for node in graph.nodes:
        in_degree = len(list(graph.in_edges(node)))  #Calcola il numero di archi entranti per il nodo v
        for b in range(0, min(k, in_degree) + 1):
            #DP[node][b] corrisponde al valore massimo della funzione λ(Dv,b), calcolato per ogni nodo v e per ciascuna dimensione b del sottoinsieme.
            DP[node][b] = compute_lambda(graph, node, log, b, p)  # Passa p come argomento, ovvero, le probabilità p(u,v) calcolate sono utilizzate per stimare la probabilità che un nodo influenzi un altro
    return DP


def greedy_allocation(graph, DP, k, p, threshold=1e-6):
    """
    Esegue l'allocazione Greedy per selezionare gli archi ottimali basati sulla tabella DP.
    """
    best_allocation = []   #Tiene traccia del numero ottimale di archi b selezionati per ciascun nodo.
    selected_edges = []   #Lista degli archi effettivamente selezionati nel grafo finale.
    total_weight = 0   #Somma totale dei valori λ associati alla selezione degli archi.

    for node in graph.nodes:
        # il miglior numero di collegamenti (b) per il nodo
        best_b = max(range(k + 1), key=lambda b: DP[node][b] if DP[node][b] is not None else -float('inf'))
        best_allocation.append(best_b)  #salva il valore ottimale b nella lista

        # Seleziona solo gli archi con probabilità p(u,v) maggiore o uguale alla soglia threshold.
        # In questo modo si eliminano gli archi con probabilità trascurabili
        in_edges = list(graph.in_edges(node, data=True))
        valid_edges = [
            (u, v, data) for u, v, data in in_edges if p.get((u, v), 0) >= threshold
        ]
        valid_edges_sorted = sorted(valid_edges, key=lambda edge: p.get((edge[0], edge[1]), 0), reverse=True)  #gli archi valiti vengono ordinati in ordine decrescente di probabilità

        # selezione dei migliori b archi
        selected_edges.extend(valid_edges_sorted[:best_b])
        #Aggiungo il valore della funzione di log-similarity λ associato alla scelta del miglior b al peso totale.
        total_weight += DP[node][best_b] if DP[node][best_b] is not None else -float('inf')

    return total_weight, best_allocation, selected_edges


#trovo il sotto-grafo ottimale che massimizza la log-verosimiglianza, considerando il vincolo sui collegamenti totali.
#

def optimal_sparse_with_probabilities(graph, k, log):
    p = compute_probabilities(graph, log)
    DP = fill_dp(graph, log, k, p)  # Passa p come argomento
    best_likelihood, _, selected_edges = greedy_allocation(graph, DP, k, p)
    return best_likelihood, selected_edges

#tipologia di grafo utilizzato per i test
def create_erdos_renyi_graph(num_nodes, edge_prob):
    G = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True)
    for u, v in G.edges:
        G[u][v]['weight'] = np.random.rand()  # Peso casuale per ogni collegamento
    return G

#tipologia di grafo utilizzato per i test
def create_barabasi_albert_graph(num_nodes, edges_per_node):
    G = nx.barabasi_albert_graph(num_nodes, edges_per_node)
    G = G.to_directed()  # Rendi il grafo orientato
    for u, v in G.edges:
        G[u][v]['weight'] = np.random.rand()  # Peso casuale per ogni collegamento
    return G

#il log è il registro delle propagazioni; una lista di azioni osservate, dove ogni azione è rappresentata come una tupla:
# (target,time,influencers); target: Il nodo influenzato ; time: Il momento dell'azione ; influencers: I nodi che possono aver influenzato il target
#creazione del log delle propagazioni secondo il modello INDIPENDENT CASCADE (IC)
def generate_log_ic_model(graph, num_actions):
    """
    Genera un log di azioni per il modello IC, rispettando la struttura del grafo.
    """
    log = []
    nodes = list(graph.nodes)

    #Questo ciclo viene ripetuto per num_actions volte, simulando ogni volta un'azione in cui un nodo target viene influenzato.
    for _ in range(num_actions):
        # Scegli un nodo bersaglio casuale
        target = np.random.choice(nodes)

        # Trova i potenziali influencer (nodi con archi in ingresso verso il target)
        influencers = [u for u, v in graph.in_edges(target)]

        if not influencers:
            continue  # Salta se non ci sono influencer per il nodo target

        # Scegli influencer casuali
        selected_influencers = np.random.choice(influencers, size=np.random.randint(1, len(influencers) + 1), replace=False)

        # np.random.randint(1, 100): un timestamp casuale per l'azione
        log.append((target, np.random.randint(1, 100), list(selected_influencers)))

    return log

"""
La funzione determine_k calcola il valore di k, che rappresenta il numero di collegamenti (archi) 
che si desidera mantenere nel grafo durante il processo di sparsificazione
"""
def determine_k(graph, sparsity_level=0.1):   #sparsity_level: un valore che rappresenta il livello di sparsità desiderato, espresso come percentuale (ad esempio, 0.1 corrisponde al 10%).

    num_edges = len(graph.edges)
    #Calcola il numero di archi da mantenere nel grafo come percentuale del numero totale di archi.
    return max(1, int(sparsity_level * num_edges))  # Almeno 1 collegamento

def plot_graph(graph, title="Grafo"):
    pos = nx.spring_layout(graph)  # Posizioni dei nodi
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()


def measure_execution_time(num_nodes, num_actions, sparsity_level):
    max_in_degrees = []  # Numero massimo di archi in ingresso
    execution_times = []  # Tempi di esecuzione

    # Prova con diversi valori di edge_prob per variare il numero di archi in ingresso
    for edge_prob in [0.1, 0.2, 0.3, 0.4, 0.5]:
        # Genera il grafo
        G = create_erdos_renyi_graph(num_nodes, edge_prob)
        # Calcola il massimo grado in ingresso
        max_in_degree = max(dict(G.in_degree()).values())
        max_in_degrees.append(max_in_degree)

        # Genera il log delle propagazioni
        log = generate_log_ic_model(G, num_actions)

        # Calcola k
        k = determine_k(G, sparsity_level)

        # Misura del tempo di esecuzione
        start_time = time.time()
        optimal_sparse_with_probabilities(G, k, log)
        end_time = time.time()

        execution_time = end_time - start_time
        execution_times.append(execution_time)

        print(f"Edge prob: {edge_prob}, Max in-degree: {max_in_degree}, Time: {execution_time:.2f}s")

    # Disegno del grafico
    plt.figure(figsize=(10, 6))
    plt.plot(max_in_degrees, execution_times, marker='o')
    plt.xlabel("Numero massimo di archi in ingresso (d)")
    plt.ylabel("Tempo di esecuzione (s)")
    plt.title("Curva temporale dell'algoritmo in funzione degli archi in ingresso")
    plt.grid(True)
    plt.show()


# Blocco principale
if __name__ == "__main__":
    num_nodes = 20
    edge_prob = 0.5
    sparsity_level = 0.3
    num_actions = 50

    # Genera il grafo iniziale
    G = create_erdos_renyi_graph(num_nodes, edge_prob)
    print(f"Numero di archi nel grafo originale: {len(G.edges)}")

    # Visualizza il grafo iniziale
    plot_graph(G, title="Grafo Iniziale")

    # Genera il log delle azioni
    log = generate_log_ic_model(G, num_actions)

    # Determina il valore di k
    k = determine_k(G, sparsity_level)

    # Calcolo del sottografo ottimale
    best_likelihood, selected_edges = optimal_sparse_with_probabilities(G, k, log)

    # Gestione del risultato
    if not selected_edges:
        print("Errore: Nessun arco significativo selezionato.")
    else:
        print(f"Log-verosimiglianza massima: {best_likelihood:.4f}")
        print(f"Numero di collegamenti selezionati: {len(selected_edges)}")
        optimal_graph = nx.DiGraph()
        optimal_graph.add_edges_from(selected_edges)

        # Visualizza il sottografo ottimale
        plot_graph(optimal_graph, title="Sottografo Ottimale")

    # Misura il tempo di esecuzione su diversi grafi
    measure_execution_time(num_nodes, num_actions, sparsity_level)
