#Optimal Sparse

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


def log_likelihood_original(graph, p, log):
    """
    Calcola la log-verosimiglianza del grafo originale usando le probabilità p(u, v) e il log delle azioni.
    """
    likelihood = 0

    # Per ogni arco nel grafo, calcola la log-verosimiglianza
    for u, v in graph.edges:
        prob = p.get((u, v), 0.5)  # Ottieni la probabilità di esistenza dell'arco
        if prob > 0:  # Evita log(0)
            likelihood += np.log(prob)

    return likelihood



def greedy_allocation(graph, DP, k, p, threshold=1e-6):
    """
    Esegue l'allocazione per selezionare esattamente k archi ottimali basati sulla tabella DP.
    """
    best_allocation = []   # Tiene traccia del numero ottimale di archi b selezionati per ciascun nodo.
    selected_edges = []    # Lista degli archi effettivamente selezionati nel grafo finale.
    total_weight = 0       # Somma totale dei valori λ associati alla selezione degli archi.

    for node in graph.nodes:
        # Il miglior numero di collegamenti (b) per il nodo
        best_b = max(range(k + 1), key=lambda b: DP[node][b] if DP[node][b] is not None else -float('inf'))
        best_allocation.append(best_b)  # Salva il valore ottimale b nella lista

        # Seleziona solo gli archi con probabilità p(u,v) maggiore o uguale alla soglia threshold.
        # In questo modo si eliminano gli archi con probabilità trascurabili
        in_edges = list(graph.in_edges(node, data=True))
        valid_edges = [
            (u, v, data) for u, v, data in in_edges if p.get((u, v), 0) >= threshold
        ]
        valid_edges_sorted = sorted(valid_edges, key=lambda edge: p.get((edge[0], edge[1]), 0), reverse=True)  # Gli archi validi vengono ordinati in ordine decrescente di probabilità

        # Seleziona solo i primi archi per rispettare il numero di archi selezionati
        selected_edges.extend(valid_edges_sorted[:best_b])

        # Limita il numero totale di archi selezionati a k
        if len(selected_edges) > k:
            selected_edges = selected_edges[:k]

        # Aggiungi il valore della funzione di log-similarity λ associato alla scelta del miglior b al peso totale.
        total_weight += DP[node][best_b] if DP[node][best_b] is not None else -float('inf')

    # Verifica che il numero di archi selezionati sia esattamente k
    if len(selected_edges) != k:
        print(f"Errore: il numero di archi selezionati ({len(selected_edges)}) non corrisponde a k ({k})")

    return total_weight, best_allocation, selected_edges

# Calcolo della log-verosimiglianza del grafo originale (senza sparsificazione)
def calculate_original_log_likelihood(graph, log, p):
    original_likelihood = 0
    for edge in graph.edges():
        # Verifica e sostituisci con una probabilità di default se la probabilità è zero
        edge_prob = p.get(edge, 1e-7)  # Imposta una probabilità di default
        if edge_prob == 0:
            edge_prob = 1e-7  # Assicurati che non sia zero
        likelihood = np.log(edge_prob)  # Calcola il logaritmo della probabilità
        original_likelihood += likelihood
    return original_likelihood


#trovo il sotto-grafo ottimale che massimizza la log-verosimiglianza, considerando il vincolo sui collegamenti totali.
def optimal_sparse_with_probabilities(graph, sparsity_level, log):
    # Calcola il valore di k in base al livello di sparsità
    k = determine_k(graph, sparsity_level)

    # Calcola le probabilità p(u, v) per il grafo
    p = compute_probabilities(graph, log)

    # Calcola la tabella DP per il grafo e le probabilità
    DP = fill_dp(graph, log, k, p)

    # Ottieni il sottografo ottimale
    best_likelihood, _, selected_edges = greedy_allocation(graph, DP, k, p)

    # Verifica che il numero di archi selezionati corrisponda a k
    if len(selected_edges) != k:
        print(f"Errore: il numero di archi selezionati ({len(selected_edges)}) non corrisponde a k ({k})")

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

def create_watts_strogatz_graph(num_nodes, k=4, p=0.3):
    """
    Crea un grafo di Watts-Strogatz con num_nodes nodi,
    k collegamenti per nodo e una probabilità di ri-collegamento p.
    """
    G = nx.watts_strogatz_graph(num_nodes, k, p)
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

        log.append((target, np.random.randint(1, 100), list(selected_influencers)))

    return log

"""
La funzione determine_k calcola il valore di k, che rappresenta il numero di collegamenti (archi) 
che si desidera mantenere nel grafo durante il processo di sparsificazione
"""
def determine_k(graph, sparsity_level=0.1):

    num_edges = len(graph.edges)
    k = int(sparsity_level * num_edges)  # Percentuale di archi da mantenere
    k = max(k, 1)  # Assicurati di mantenere almeno 1 arco
    return k

def plot_graph(graph, selected_edges=None, title="Graph"):
    """
    Visualizza un grafo utilizzando NetworkX e Matplotlib,
    con gli archi selezionati evidenziati in rosso.
    """
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(graph, seed=42)  # Posizioni dei nodi

    # Disegna i nodi
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=500)

    # Disegna tutti gli archi in grigio
    nx.draw_networkx_edges(graph, pos, edge_color="gray", alpha=0.5)

    # Evidenzia gli archi selezionati in rosso
    if selected_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=selected_edges,
            edge_color="red",
            width=2,
            alpha=0.8,
            label="Selected Edges"
        )

    # Disegna le etichette dei nodi
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black")

    # Aggiungi titolo e legenda
    plt.title(title)
    plt.legend()
    plt.show()


def compare_graph_types_with_sparsity():
    """
    Confronta il comportamento dell'algoritmo su diverse tipologie di grafi
    e livelli di sparsità. Visualizza sia i grafi originali che quelli sparsificati.
    """
    graph_types = {
        "Erdős-Rényi": lambda n: create_erdos_renyi_graph(n, edge_prob=0.3),
        "Barabási-Albert": lambda n: create_barabasi_albert_graph(n, edges_per_node=3),
        "Watts-Strogatz": lambda n: nx.watts_strogatz_graph(n, k=4, p=0.3)
    }

    num_nodes = 30
    num_actions = 50
    sparsity_levels = [0.1, 0.2, 0.4, 0.8]  # Diversi livelli di sparsità

    results = []

    for graph_name, graph_fn in graph_types.items():
        # Crea il grafo originale
        G = graph_fn(num_nodes)
        if graph_name == "Watts-Strogatz":
            G = G.to_directed()  # Rendi Watts-Strogatz orientato
        print(f"\nTesting {graph_name} graph with {len(G.edges)} edges.")

        # Genera il log delle azioni
        log = generate_log_ic_model(G, num_actions)

        # Calcola le probabilità p(u, v) per il grafo
        p = compute_probabilities(G, log)

        # Calcola la log-verosimiglianza del grafo originale (prima di modificare il grafo)
        original_likelihood = calculate_original_log_likelihood(G, log, p)
        print(f"Log-Likelihood del grafo originale: {original_likelihood:.4f}")

        # Visualizza il grafo originale
        plot_graph(G, title=f"{graph_name} - Original")

        for sparsity_level in sparsity_levels:
            print(f"\nTesting sparsity level: {sparsity_level}")

            # Genera il log delle azioni
            log = generate_log_ic_model(G, num_actions)

            # Determina il valore di k per questo livello di sparsità
            k = determine_k(G, sparsity_level)

            # Calcola il sottografo ottimale per questo livello di sparsità
            best_likelihood, selected_edges = optimal_sparse_with_probabilities(G, sparsity_level, log)

            # Visualizza il grafo sparsificato
            plot_graph(G, selected_edges, title=f"{graph_name} - Sparsified (Sparsity = {sparsity_level})")

            # Aggiungi i risultati
            results.append({
                "Graph Type": graph_name,
                "Sparsity Level": sparsity_level,
                "Edges": len(G.edges),
                "Selected Edges": len(selected_edges),
                "Log-Likelihood (Original)": original_likelihood,
                "Log-Likelihood (Sparsified)": best_likelihood
            })

    # Stampa i risultati
    for result in results:
        print(f"\nGraph Type: {result['Graph Type']}, Sparsity Level: {result['Sparsity Level']}")
        print(f"  Original Edges: {result['Edges']}")
        print(f"  Selected Edges: {result['Selected Edges']}")
        print(f"  Log-Likelihood (Original): {result['Log-Likelihood (Original)']:.4f}")
        print(f"  Log-Likelihood (Sparsified): {result['Log-Likelihood (Sparsified)']:.4f}")

def plot_execution_time():
    """
    Misura i tempi di esecuzione su grafi di diverse dimensioni e livelli di sparsità.
    Ogni livello di sparsità sarà rappresentato da una linea distinta nel grafico.
    Ora viene misurato il tempo di esecuzione una sola volta per ogni configurazione.
    """
    #node_counts = range(10, 80, 2)  # Numero di nodi da 10 a 30 con passo di 2
    node_counts = [5,10,20,40,80,160,320,640,1280,2560,5120,10240,20480]
    sparsity_levels = [0.1, 0.2, 0.4, 0.8]  # I vari livelli di sparsità da testare
    all_times = {sparsity: [] for sparsity in sparsity_levels}  # Dizionario per memorizzare i tempi per ogni sparsità

    for sparsity_level in sparsity_levels:
        times = []  # Lista per memorizzare i tempi di esecuzione per un livello di sparsità

        for num_nodes in node_counts:
            # Creazione di un grafo di Watts-Strogatz
            G = create_watts_strogatz_graph(num_nodes, k=4, p=0.3)

            # Misura il tempo di esecuzione per una sola iterazione
            start_time = time.time()
            log = generate_log_ic_model(G, num_actions=10)  # Aggiungi la generazione del log
            best_likelihood, selected_edges = optimal_sparse_with_probabilities(G, sparsity_level, log)
            end_time = time.time()

            # Calcola il tempo di esecuzione per questa configurazione
            execution_time = end_time - start_time
            times.append(execution_time)
            print(f"Sparsity: {sparsity_level}, Nodes: {num_nodes}, Time: {execution_time:.4f}s")

        all_times[sparsity_level] = times

    # Grafico dei tempi di esecuzione per diversi livelli di sparsità
    plt.figure(figsize=(10, 6))

    # Disegna una linea per ciascun livello di sparsità
    for sparsity_level, times in all_times.items():
        plt.plot(node_counts, times, marker='o', label=f'Sparsity = {sparsity_level}')

    plt.title('Execution Time vs. Number of Nodes for Different Sparsity Levels (Watts-Strogatz)')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Blocco principale
if __name__ == "__main__":
    num_nodes = 20
    edge_prob = 0.5
    num_actions = 50

    # Genera il grafo iniziale
    G = create_erdos_renyi_graph(num_nodes, edge_prob)
    print(f"Numero di archi nel grafo originale: {len(G.edges)}")

    # Visualizza il grafo iniziale
    plot_graph(G, title="Grafo Iniziale")

    # Genera il log delle azioni
    log = generate_log_ic_model(G, num_actions)

    # Ciclo per testare diversi livelli di sparsità
    sparsity_levels = [0.1, 0.2, 0.4, 0.8]
    for sparsity_level in sparsity_levels:
        print(f"\nTesting sparsity level: {sparsity_level}")

        # Determina il valore di k per questo livello di sparsità
        k = determine_k(G, sparsity_level)
        print(f"Numero di archi da mantenere (k): {k}")

        # Calcolo del sottografo ottimale per questo livello di sparsità
        best_likelihood, selected_edges = optimal_sparse_with_probabilities(G, sparsity_level, log)

        # Gestione del risultato
        if not selected_edges:
            print("Errore: Nessun arco significativo selezionato.")
        else:
            print(f"Log-verosimiglianza massima: {best_likelihood:.4f}")
            print(f"Numero di collegamenti selezionati: {len(selected_edges)} (Atteso: {k})")
            optimal_graph = nx.DiGraph()
            optimal_graph.add_edges_from(selected_edges)

            # Visualizza il sottografo ottimale
            plot_graph(G, selected_edges, title=f"Grafo Ottimale (Sparsity = {sparsity_level})")

    # Esegui il confronto con diverse tipologie di grafi
    compare_graph_types_with_sparsity()

    # Visualizza il grafico dei tempi di esecuzione
    plot_execution_time()

