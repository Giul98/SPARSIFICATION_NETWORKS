import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import numpy as np
import matplotlib


def plot_graph(graph, selected_edges=None, title="Graph"):   #un insieme di archi evidenziati in rosso per distinguere gli archi selezionati da quelli non selezionati.
    """
    Visualizza un grafo utilizzando NetworkX e Matplotlib.
    """
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(graph, seed=42)  # Posizioni dei nodi
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)  # Posizioni dei nodi

    if selected_edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=selected_edges, edge_color="red", width=2, alpha=0.8, label="Selected Edges"
        )

    plt.title(title)
    plt.legend()
    plt.show()  # Mostra la finestra grafica


def compute_probabilities(graph, log, max_iterations=100, tol=1e-5):
    """"
    Calcola iterativamente le probabilità p(u,v) per ogni arco nel grafo utilizzando un processo iterativo basato su un log di azioni.

    """
    p = {edge: 0.5 for edge in graph.edges}  # Inizializza p(u, v)
    p_prev = p.copy()  #dizionario con le probabilitò aggiornate per ogni arco

    for _ in range(max_iterations):
        updates = {}   #In ogni iterazione, un dizionario updates memorizza le nuove probabilità per gli archi.
        for u, v in graph.edges:
            A_plus = [action for action in log if v == action[0] and u in action[2]]  #azioni in cui u influenza v (A+)
            A_minus = [action for action in log if v == action[0] and u not in action[2]]  # azioni in cui u non influenza v (A-)

            #Calcolo di P+
            # Per ogni azione in A_plus, considera la probabilità che v sia stato influenzato da u, escludendo gli altri nodi w.
            P_plus_sum = sum(
                1 - np.prod([(1 - p.get((w, v), 0)) for w in action[2] if w != u])
                for action in A_plus
            )

            #Calcolo di P-
            # Per ogni azione in A_minus, considera la probabilità che v non sia stato influenzato dai nodi w, escludendo u.

            P_minus_prod = np.prod(
                np.prod([(1 - p.get((w, v), 0)) for w in action[2] if w != u])
                for action in A_minus
            )

            denom = len(A_plus) + len(A_minus)
            if denom > 0:
                updates[(u, v)] = P_plus_sum / denom    #calcolo di p(u,v)

        max_diff = max(abs(updates[e] - p_prev[e]) for e in updates) #calcola la differenza massima tra le probabilità dell'iterazione attuale e quelle dell'iterazione precedente
        p_prev.update(updates)
        if max_diff < tol:
            break

    return p_prev   #Restituisce il dizionario p_prev contenente le probabilità aggiornate p(u,v) per ogni arco


def log_likelihood(graph, p, log):
    """
    Calcola la log-verosimiglianza di ogni arco nel grafo basandosi sul log delle azioni e sulle probabilità calcolate.
    """
    likelihood = {}
    for u, v in graph.edges:
        likelihood[(u, v)] = 0
        for action in log:
            target, _, influencers = action
            if target != v:
                continue

            # Probabilità che u influenzi v
            P_plus = 1 - np.prod([(1 - p.get((w, v), 0)) for w in influencers])
            # Probabilità che v non sia influenzato dagli altri nodi, escluso u
            P_minus = np.prod([(1 - p.get((w, v), 0)) for w in influencers if w != u])

            # Correzione: aggiungi un termine di regolarizzazione per evitare log(0)
            P_plus = max(P_plus, 1e-7)  # evitare log(0)
            P_minus = max(P_minus, 1e-7)  # evitare log(0)

            # Log-likelihood per l'arco
            likelihood[(u, v)] += np.log(P_plus) + np.log(P_minus)  # somma dei log

    return likelihood


def spine_algorithm(graph, k, p, log):
    """
    Implementazione Spine Algorithm basata su log-likelihood.
    Seleziona k archi del grafo con la massima log-verosimiglianza.
    """
    # Calcola la log-likelihood per ogni arco
    arc_likelihoods = log_likelihood(graph, p, log)  #Usa la funzione log_likelihood per calcolare la log-verosimiglianza di ogni arco.

    # Ordina gli archi in base alla loro log-likelihood decrescente
    sorted_arcs = sorted(arc_likelihoods.items(), key=lambda x: x[1], reverse=True)

    # Seleziona i primi k archi
    selected_edges = set()
    for arc, likelihood in sorted_arcs[:k]:
        selected_edges.add(arc)

    return selected_edges

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

def measure_execution_time():
    """
    Misura i tempi di esecuzione del Spine Algorithm su grafi di diverse dimensioni con maggiore granularità.
    """
    # range di nodi 10, 20, 40, 80, 160, 320, 640, 1280, 2560
    node_counts = [10, 20, 40, 80, 160, 320, 640, 1280, 2560,5120,10240,20480]  # Nodi da analizzare
    sparsity_levels = [0.1, 0.2, 0.4, 0.8]  # Livelli di sparsità
    times = []

    for num_nodes in node_counts:
        for sparsity_level in sparsity_levels:
            # Creazione di un grafo casuale
            #G = nx.erdos_renyi_graph(num_nodes, p=0.3, directed=True)
            G = create_watts_strogatz_graph(num_nodes, k=4, p=0.3)
            k = int(sparsity_level * len(G.edges))

            # Misura più volte per ridurre la varianza
            total_time = 0
            iterations = 5  # Esegui la misura 5 volte e fai la media
            for _ in range(iterations):
                start_time = time.time()
                selected_edges = spine_algorithm(G, k, p, log)
                end_time = time.time()
                total_time += (end_time - start_time)

            # Calcola la media del tempo
            avg_time = total_time / iterations
            times.append((num_nodes, sparsity_level, avg_time))
            print(f"Nodes: {num_nodes}, Sparsity: {sparsity_level}, Avg Time: {avg_time:.4f}s")

    return times


def test_scalability():
    # Genera un grafo di dimensioni crescenti e valuta i tempi di esecuzione
    node_counts = [10,20,40,80,160,320,640,1280]
    edge_prob = 0.2

    for num_nodes in node_counts:
        G = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True)
        k = int(0.3 * len(G.edges))  # Esempio: manteniamo il 30% degli archi

        start_time = time.time()
        selected_edges = spine_algorithm(G, k, p, log)
        end_time = time.time()

        print(f"Nodes: {num_nodes}, Edges: {len(G.edges)}, Time: {end_time - start_time:.4f}s")

def generate_log_ic_model(graph, num_actions):
    """
    Genera un log di azioni per il modello IC, rispettando la struttura del grafo.
    Ogni azione è rappresentata come una tupla:
    (nodo bersaglio, timestamp, influenzatori).
    """
    log = []
    nodes = list(graph.nodes)

    for _ in range(num_actions):
        # Scegli un nodo bersaglio casuale
        target = random.choice(nodes)

        # Trova i potenziali influencer (nodi con archi in ingresso verso il target)
        influencers = [u for u, v in graph.in_edges(target)]

        if not influencers:
            continue  # Salta se non ci sono influencer per il nodo target

        # Scegli un sottoinsieme casuale di influencer
        selected_influencers = random.sample(influencers, k=random.randint(1, len(influencers)))

        # Aggiungi l'azione al log (target, timestamp casuale, lista degli influenzatori)
        log.append((target, random.randint(1, 100), selected_influencers))

    return log


def compare_graph_types_spine():
    """
    Confronto del comportamento di Spine su diversi tipi di grafi, visualizzando il grafo originale
    seguito dai grafi sparsificati con i livelli di sparsità richiesti, calcolando la log-likelihood.
    """
    graph_types = {
        "Erdős-Rényi": lambda n: nx.erdos_renyi_graph(n, p=0.3, directed=True),
        "Barabási-Albert": lambda n: nx.barabasi_albert_graph(n, m=3).to_directed(),
        "Watts-Strogatz": lambda n: nx.watts_strogatz_graph(n, k=4, p=0.3).to_directed(),
    }

    sparsity_levels = [0.1, 0.2, 0.4, 0.8]  # Livelli di sparsità
    num_nodes = 50  #  test su un grafo di 50 nodi
    num_actions = 100  # Numero di azioni nel log

    results = []

    for graph_name, graph_fn in graph_types.items():
        print(f"Testing {graph_name} graph with {num_nodes} nodes.")

        # Genera il grafo originale
        G = graph_fn(num_nodes)
        print(f"Numero di archi originali: {len(G.edges)}")

        # Visualizza il grafo originale
        plot_graph(G, title=f"{graph_name} - Grafo Originale")

        # Calcola il log-likelihood per il grafo originale (massimo per il grafo completo)
        log = generate_log_ic_model(G, num_actions)  # Genera il log delle azioni
        p = compute_probabilities(G, log)  # Calcola le probabilità
        original_likelihood = log_likelihood(G, p, log)
        original_log_likelihood = sum(original_likelihood.values())  # Log-likelihood massima per il grafo originale

        for sparsity_level in sparsity_levels:
            # Determina il numero di archi da mantenere (k)
            k = int(sparsity_level * len(G.edges))

            # Calcolo dei grafi sparsificati
            selected_edges = spine_algorithm(G, k, p, log)

            # Calcola la log-likelihood per il grafo sparsificato (solo per gli archi selezionati)
            sparsified_likelihood = {edge: original_likelihood[edge] for edge in selected_edges}
            sparsified_log_likelihood = sum(sparsified_likelihood.values())

            # Visualizza il grafo sparsificato
            plot_graph(G, selected_edges, title=f"{graph_name} - Grafo Sparsificato (Sparsity {sparsity_level:.1f})")

            results.append({
                "Graph Type": graph_name,
                "Selected Edges": len(selected_edges),
                "Sparsified Log-Likelihood": sparsified_log_likelihood,
                "Original Log-Likelihood": original_log_likelihood
            })

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)

    df_results = pd.DataFrame(results)
    df_results = df_results[["Graph Type", "Selected Edges", "Sparsified Log-Likelihood", "Original Log-Likelihood"]]

    print(df_results)
    return df_results



def determine_k(graph, sparsity_level=0.1):
    """
    Determina il valore di k, il numero massimo di archi da mantenere.
    """
    num_edges = len(graph.edges)
    return max(1, int(sparsity_level * num_edges))  # Almeno 1 arco

def plot_execution_times(times):
    """
    Disegna la curva temporale computazionale per dimostrare la complessità logaritmica.
    """
    # Converte i dati in un DataFrame per facilitare l'elaborazione
    df = pd.DataFrame(times, columns=['Nodes', 'Sparsity', 'Time'])

    plt.figure(figsize=(10, 6))
    for sparsity in sorted(df['Sparsity'].unique()):
        subset = df[df['Sparsity'] == sparsity]
        plt.plot(
            subset['Nodes'],
            subset['Time'],
            marker='o',
            label=f'Sparsity {sparsity:.1f}'
        )


    plt.yscale('log')

    # Dettagli del grafico
    plt.title('Curva Temporale dell\'Algoritmo (Scala Logaritmica)')
    plt.xlabel('Numero di Nodi')
    plt.ylabel('Tempo di Esecuzione (s)')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend(title='Livello di Sparsità')
    plt.show()

if __name__ == "__main__":
    # Parametri iniziali
    num_nodes = 100
    edge_prob = 0.3
    sparsity_level = 0.3
    num_actions = 50

    # Generazione del grafo
    G = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True)
    print(f"Numero di nodi: {num_nodes}, Numero di archi: {len(G.edges)}")

    # Genera il log delle azioni
    log = generate_log_ic_model(G, num_actions)  # Definizione aggiunta sopra

    # Calcola le probabilità
    p = compute_probabilities(G, log)

    # Determina il numero massimo di archi da selezionare
    k = int(sparsity_level * len(G.edges))
    print(f"Numero di archi da selezionare (k): {k}")

    # Visualizza il grafo iniziale
    plot_graph(G, title="Grafo Iniziale")

    # Sparsifica il grafo
    selected_edges = spine_algorithm(G, k, p, log)

    # Verifica che siano stati selezionati archi
    if selected_edges:
        print(f"Numero di archi selezionati: {len(selected_edges)}")
        print(f"Archi selezionati: {selected_edges}")

        # Visualizza il grafo sparsificato
        plot_graph(G, selected_edges, title="Grafo Sparsificato")
    else:
        print("Errore: Nessun arco significativo selezionato.")



    #confronta spine su diversi grafi
    compare_graph_types_spine()

    # Misura i tempi di esecuzione
    print("\nMisurazione dei tempi di esecuzione:")
    execution_times = measure_execution_time()
    plot_execution_times(execution_times)

    # Visualizza i tempi di esecuzione
    #plot_execution_times(execution_times)


    # Testa la scalabilità
    print("\nTest di scalabilità:")
    test_scalability()