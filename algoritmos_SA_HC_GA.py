"""
Traveling Salesman Problem (TSP) Solvers

This script provides implementations of various algorithms to solve the Traveling Salesman Problem (TSP), 
an optimization challenge where the goal is to find the shortest route visiting all cities exactly once and returning to the origin.

Algorithms included:
- Simulated Annealing (SA)
- Hill Climbing (HC)
- Genetic Algorithm (GA)

Each algorithm explores a different approach to optimize the route. This script also includes helper functions 
for data normalization, random solution generation, and cost calculation.
"""

# Manipulação de dados
import attr
import numpy as np
import pandas as pd
import math
# Geração de números aleatórios
import random
from math import sqrt

# Geração de gráficos
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Funções estatísticas
from scipy.special import softmax


# Normalização de dados
def normalizacao(dados):
    return (dados / np.max(dados)).tolist()

# Cria uma solucao inicial com as cidades em um ordem aleatoria
def solucao_aleatoria(tsp):
    cidades = list(tsp.keys())
    solucao = []

    # as 3 linhas abaixo não são estritamente necessarias, servem
    # apenas para fixar a primeira cidade da lista na solução
    cidade = cidades[0]
    solucao.append(cidade)
    cidades.remove(cidade)

    for _ in range(0,len(cidades)):
        #print(_, cidades, solucao)
        cidade = random.choice(cidades)

        solucao.append(cidade)
        cidades.remove(cidade)

    return solucao


# Função Objetivo: calcula custo de uma dada solução.
# Obs: Neste caso do problema do caixeiro viajante (TSP problem),
# o custo é o comprimento da rota entre todas as cidades.
def calcula_custo(tsp, solucao):

    N = len(solucao)
    custo = 0

    for i in range(N):

        # Quando chegar na última cidade, será necessário
        # voltar para o início para adicionar o
        # comprimento da rota da última cidade
        # até a primeira cidade, fechando o ciclo.
        #
        # Por isso, a linha abaixo:
        k = (i+1) % N
        cidadeA = solucao[i]
        cidadeB = solucao[k]

        custo += tsp.loc[cidadeA, cidadeB]

        #print(tsp.loc[cidadeA, cidadeB], cidadeA,cidadeB)

    return custo

"""### Gera Vizinhos

Obs: a função `obtem_vizinhos` descrita abaixo foi gerada de forma simplificada, pois ela assume que todos os vizinhos possuem rota direta entre si. Isto tem caráter didático para simplifcar a solução. Observe que na prática isso nem sempre existe rotas diretas entre todas as cidades e, em tais casos, pode ser necessário modificar a função para corresponder a tais restrições.
"""

# A partir de uma dada solução, gera diversas variações (vizinhos)
def gera_vizinhos(solucao):

    N = len(solucao)
    for i in range(1, N):       # deixa o primeiro fixo
        for j in range(i + 1, N):
            vizinho = solucao.copy()
            vizinho[i] = solucao[j]
            vizinho[j] = solucao[i]

            yield(vizinho)

"""### Seleciona Melhor Vizinho"""

def obtem_melhor_vizinho(tsp, solucao):
    melhor_custo = calcula_custo(tsp, solucao)
    melhor_vizinho = solucao

    for vizinho in gera_vizinhos(solucao):
        custo_atual = calcula_custo(tsp, vizinho)
        if custo_atual < melhor_custo:
            melhor_custo = custo_atual
            melhor_vizinho = vizinho

    return melhor_vizinho, melhor_custo

"""### Random-Walk - clássico"""

def obtem_vizinho_aleatorio(tsp, solucao):

    vizinhos = list(gera_vizinhos(solucao))

    aleatorio_vizinho  = random.choice(vizinhos)
    aleatorio_custo    = calcula_custo(tsp, aleatorio_vizinho)

    return aleatorio_vizinho, aleatorio_custo



# distancia Euclidiana entre dois pontos
def distancia(x1,y1,x2,y2):
    dx = x2 - x1
    dy = y2 - y1
    return sqrt(dx**2 + dy**2)

# Calcula matriz de distancias.
#
# OBS:  Não é estritamente necessario calculá-las a priori.
#       Foi feito assim apenas para fins didáticos.
#       Ao invés, as distâncias podem ser calculadas sob demanda.

def gera_matriz_distancias(Coordenadas):

    n_cidades = len(Coordenadas)
    dist = np.zeros((n_cidades,n_cidades), dtype=float)

    for i in range(0, n_cidades):
        for j in range(i+1, n_cidades):
            x1,y1 = Coordenadas.iloc[i]
            x2,y2 = Coordenadas.iloc[j]

            dist[i,j] = distancia(x1,y1,x2,y2)
            dist[j,i] = dist[i,j]

    return dist

"""### Gerador de Problemas Aleatórios"""

# Gera aleatoriamente as coordenadas de N cidades.
# Obs: esta informação geralmente é fornecida como entrada do problema.

def gera_coordenadas_aleatorias(n_cidades):
    minimo = 10
    maximo = 590
    escala = (maximo-minimo)-1

    # gera n coordenadas (x,y) aleatorias entre [min, max]
    X = minimo + escala * np.random.rand(n_cidades)
    Y = minimo + escala * np.random.rand(n_cidades)
    coordenadas = {'X':X, 'Y': Y}

    cidades = ['A'+str(i) for i in range(n_cidades)]

    df_cidades = pd.DataFrame(coordenadas, index=cidades)
    df_cidades.index.name = 'CIDADE'

    return df_cidades

# Recebe uma lista com as coordenadas reais de uma cidade e
# gera uma matriz de distancias entre as cidades.
# Obs: a matriz é simetrica e com diagonal nula
def gera_problema_tsp(df_cidades):
    # nomes ficticios das cidades
    cidades = df_cidades.index

    # calcula matriz de distancias
    distancias = gera_matriz_distancias(df_cidades)

    # cria estrutura para armazena as distâncias entre todas as cidades
    tsp = pd.DataFrame(distancias, columns=cidades, index=cidades)

    return tsp


# Plota a solução do roteamento das cidades
# usando a biblioteca PLOTLY
def plota_rotas(df_cidades, ordem_cidades):
    df_solucao = df_cidades.copy()
    df_solucao = df_solucao.reindex(ordem_cidades)

    X = df_solucao['X']
    Y = df_solucao['Y']
    cidades = list(df_solucao.index)

    # cria objeto gráfico
    fig = go.Figure()

    fig.update_layout(autosize=False, width=500, height=500, showlegend=False)

    # gera linhas com as rotas da primeira ate a ultima cidade
    fig.add_trace(go.Scatter(x=X, y=Y,
                             text=cidades, textposition='bottom center',
                             mode='lines+markers+text',
                             name=''))

    # acrescenta linha da última para a primeira para fechar o ciclo
    fig.add_trace(go.Scatter(x=X.iloc[[-1,0]], y=Y.iloc[[-1,0]],
                             mode='lines+markers', name=''))

    fig.show()

"""### Boxplots"""

def boxplot_sorted(df, rot=90, figsize=(12,6), fontsize=20):
    df2 = df.T
    meds = df2.median().sort_values(ascending=False)
    axes = df2[meds.index].boxplot(figsize=figsize, rot=rot, fontsize=fontsize,
                                   boxprops=dict(linewidth=4, color='cornflowerblue'),
                                   whiskerprops=dict(linewidth=4, color='cornflowerblue'),
                                   medianprops=dict(linewidth=4, color='firebrick'),
                                   capprops=dict(linewidth=4, color='cornflowerblue'),
                                   flierprops=dict(marker='o', markerfacecolor='dimgray',
                                        markersize=12, markeredgecolor='black'),
                                   return_type="axes")

    axes.set_title("Cost of Algorithms", fontsize=fontsize)


### Executa N vezes - ESTRUTURADA com DataFrame
"""
a seguir, é apresentada uma forma mais estruturada de se rodar várias vezes usando a estrutura de dados **`DataFrame`** para armazenar os resultados e permitir visualização de box-plots
"""

# Cria estruta de dados (DataFrame) para armazenar vários resultados
# diferentes e visualizá-los através de estatísticas

def cria_df_custos(algoritmos, n_vezes):

    nomes_algoritmos  = algoritmos.keys()

    n_lin = len(nomes_algoritmos)
    n_col = n_vezes

    df_results = pd.DataFrame(np.zeros((n_lin,n_col)),
                              index=nomes_algoritmos)
    df_results.index.name='ALGORITMO'

    return df_results

def nomes_algs(algoritmos, **kwargs):
    nomes_algoritmos = {}
    for algoritmo in algoritmos:
        if algoritmo not in kwargs:
            nomes_algoritmos[algoritmo]={}
        for key in kwargs:
            for config in kwargs[key]:
                if len(config):
                    nomes_algoritmos[f"{algoritmo}{config}"]={}
                else:
                    nomes_algoritmos[algoritmo]={}
    return nomes_algoritmos

# Executa N vezes para gerar estatísticas da variável custo
def executa_n_vezes(tsp, algoritmos, n_vezes, **kwargs):
    # Cria DataFrame para armazenar os resultados
    nomes_algoritmos = nomes_algs(algoritmos, **kwargs)
    df_custo = cria_df_custos(nomes_algoritmos, n_vezes)
    evolucoes = {}
    evolucoes_melhores = {}

    for algoritmo, funcao_algoritmo in algoritmos.items():

        print(f"Algoritmo: {algoritmo}")

        if algoritmo not in kwargs:
            custo_melhor, solucao_melhor, evol_melhor = funcao_algoritmo(tsp)
            evolucoes[f"{algoritmo}"] = []
            for i in range(n_vezes):
                custo, solucao, evolucao = funcao_algoritmo(tsp)
                df_custo.loc[algoritmo,i] = custo
                print(f'{custo:10.3f}  {solucao}')
                if custo < custo_melhor:
                    evol_melhor=evolucao
            evolucoes_melhores[f"{algoritmo}"] = evol_melhor
        else:
            for key in kwargs:
                for config in kwargs[key]:
                    if len(config):

                        print(f"Config: {config}")

                        custo_melhor, solucao_melhor, evol_melhor = funcao_algoritmo(
                            tsp, *kwargs[algoritmo][config]
                        )
                        evolucoes[f"{algoritmo}{config}"] = []

                        for i in range(n_vezes):
                            custo, solucao, evolucao = funcao_algoritmo(
                                tsp, *kwargs[algoritmo][config]
                            )

                            evolucoes[f"{algoritmo}{config}"].append(evolucao)

                            df_custo.loc[f"{algoritmo}{config}",i] = custo
                            print(f'{custo:10.3f}  {solucao}')

                            if custo < custo_melhor:
                                evol_melhor=evolucao

                        evolucoes_melhores[f"{algoritmo}{config}"] = evol_melhor

                    else:
                        custo_melhor, solucao_melhor, evol_melhor = funcao_algoritmo(tsp)
                        evolucoes[f"{algoritmo}"] = []
                        for i in range(n_vezes):
                            custo, solucao, evolucao = funcao_algoritmo(tsp)
                            df_custo.loc[algoritmo,i] = custo
                            print(f'{custo:10.3f}  {solucao}')
                            if custo < custo_melhor:
                                evol_melhor=evolucao
                        evolucoes_melhores[f"{algoritmo}"] = evol_melhor
    return df_custo, evolucoes, evolucoes_melhores


def hill_climbing_randomrestart(tsp):
    lista_custos_hc = []
    melhor_custo_global = 0
    melhor_solucao_global = []
    #Após achar uma solução, retorna o processo e guarda a melhor solução encontrada
    for _ in range(50):
        # solucao inicial
        solucao_inicial = solucao_aleatoria(tsp)
        # melhor solucao ate o momento
        solucao_melhor, custo_melhor = obtem_melhor_vizinho(tsp, solucao_inicial)
        lista_custos_hc.append(custo_melhor)

        while True:

            # tenta obter um candidato melhor
            candidato_atual, custo_atual = obtem_melhor_vizinho(tsp, solucao_melhor)
            #print(custo_melhor, custo_atual)

            if custo_atual < custo_melhor:
                custo_melhor   = custo_atual
                solucao_melhor = candidato_atual
            else:
                if melhor_custo_global == 0:
                    melhor_custo_global = custo_melhor
                    melhor_solucao_global = solucao_melhor
                    
                if custo_melhor < melhor_custo_global:
                    melhor_custo_global = custo_melhor
                    melhor_solucao_global = solucao_melhor
                break   # custo nao melhorou, entao sai do while

            lista_custos_hc.append(custo_melhor)

    return melhor_custo_global, melhor_solucao_global, lista_custos_hc

# Funçõe auxiliares do SA

def generate_neighbor(route):
    new_route = route.copy()
    index_a = random.randint(0, len(route) - 1)
    index_b = random.randint(0, len(route) - 1)
    new_route[index_a], new_route[index_b] = new_route[index_b], new_route[index_a]
    return new_route

def acceptance_probability(current_distance, new_distance, temperature):
    if new_distance < current_distance:
        return 1.0
    else:
        return math.exp((current_distance - new_distance) / temperature)

def simulated_annealing(cities):
    initial_temperature = 1000.0
    cooling_rate = 0.993
    iterations = 1000
    lista_custos_sa = []


    distance_matrix = cities
    current_route = list(cities.keys())
    best_route = current_route.copy()
    current_distance = calcula_custo(distance_matrix,current_route)
    best_distance = current_distance
    lista_custos_sa.append(best_distance)

    temperature = initial_temperature

    #-----------------------------------------------
    iteration_list = []
    best_distances = []
    distance_list  = []
    accept_p_list  = []
    temperat_list  = []
    #-----------------------------------------------

    for iteration in range(iterations):
        new_route = generate_neighbor(current_route)
        new_distance = calcula_custo(distance_matrix, new_route)

        acceptance_prob = acceptance_probability(current_distance, new_distance, temperature)

        if random.random() < acceptance_prob:
            current_route = new_route
            current_distance = new_distance

        if new_distance < best_distance:
            best_route = new_route
            best_distance = new_distance

        lista_custos_sa.append(best_distance)
        temperature *= cooling_rate

        #-----------------------------------------------
        iteration_list += [iteration]
        best_distances += [best_distance]
        distance_list  += [current_distance]
        accept_p_list  += [acceptance_prob]
        temperat_list  += [temperature]


    return best_distance, best_route, lista_custos_sa

"""### Genetic Algorithm - clássico"""

def fitness_fn(tsp, solucoes):
    return [calcula_custo(tsp, solucao) for solucao in solucoes]

def selecao_torneio(
    populacao,
    pop_fitness,
    tamanho_torneio=2,
    sel_estocastica=[],
    luta_estocastica=False
):
    proba=None
    if len(sel_estocastica):
        proba=sel_estocastica

    idx = np.random.choice(
        range(len(populacao)),
        size=tamanho_torneio,
        replace=False,
        p=proba
    ).tolist()
    torneio = [pop_fitness[idx[0]], pop_fitness[idx[1]]]

    if luta_estocastica:
        melhor_individuo = populacao[idx[torneio.index(
            random.choices(torneio, k=1, weights=softmax(normalizacao(torneio)))
        )]]
    else:
        melhor_individuo = populacao[idx[torneio.index(
            min(torneio)
        )]]

    return melhor_individuo

def mutacao(individuo):
    if random.random() < 0.15:
        i, j = random.sample(range(len(individuo)), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]

def order1_crossover(individuo1, individuo2):
    tam = len(individuo1)
    inicio = random.randint(0, tam - 1)
    fim = random.randint(inicio, tam)
    index = [i for i in range(fim, tam)] + [i for i in range(fim)]

    filho_1 = [individuo1[i] if inicio <= i < fim else -1 for i in range(tam)]
    filho_2 = [individuo2[i] if inicio <= i < fim else -1 for i in range(tam)]

    restante_1 = [individuo2[i] for i in index if individuo2[i] not in filho_1]
    restante_2 = [individuo1[i] for i in index if individuo1[i] not in filho_2]

    j = 0
    for i in index[:(tam-fim)+(inicio)]:
        filho_1[i], filho_2[i] = restante_1[j], restante_2[j]
        j += 1
    return filho_1, filho_2

def genetic_algorithm(
    tsp,
    N=400, # Número de gerações
    K=20,   # Tamanho da população
    tamanho_torneio=2,
    escolha_estocastica=False,
    luta_estocastica=False
):
    evolucao = []
    kwargs = {
        "tamanho_torneio":tamanho_torneio,
        "sel_estocastica":[],
        "luta_estocastica":luta_estocastica
    }

    populacao = [solucao_aleatoria(tsp) for _ in range(K)]
    pop_fitness = fitness_fn(tsp, populacao)
    evolucao.append(min(pop_fitness))
    total_fit = sum(pop_fitness)

    for i in range(1, N):
        if escolha_estocastica:
            kwargs["sel_estocastica"] = softmax(normalizacao(pop_fitness))

        nova_populacao = []

        for j in range(K):
            individuo_1 = selecao_torneio(populacao, pop_fitness, **kwargs)
            individuo_2 = selecao_torneio(populacao, pop_fitness, **kwargs)
            filho_1, filho_2 = order1_crossover(individuo_1, individuo_2)
            mutacao(filho_1)
            mutacao(filho_2)
            nova_populacao.extend([filho_1, filho_2])

        populacao = nova_populacao
        pop_fitness = fitness_fn(tsp, populacao)
        evolucao.append(min(pop_fitness))
        total_fit = sum(pop_fitness)


    custo_melhor = min(pop_fitness)
    solucao_melhor = populacao[pop_fitness.index(min(pop_fitness))]

    return custo_melhor, solucao_melhor, evolucao

n_cidades = 10
df_coordenadas_aleatorias = gera_coordenadas_aleatorias(n_cidades)
tsp_aleatorio = gera_problema_tsp(df_coordenadas_aleatorias)
# tsp_aleatorio.to_csv('tsp_aleatorio.csv')
# df_coordenadas_aleatorias.to_csv('df_coordenadas_aleatorias.csv')


# Para rodar tudo junto:

algoritmos = {
    #'Random Walk - classic': solucao_aleatoria,
    # 'Hill-Climbing': hill_climbing,
    # 'Hill-Climbing - stochastic': hill_climbing_stochastic,
    # 'Hill-Climbing - first-choice': hill_climbing_firstchoice,
    'Hill-Climbing - random-restart': hill_climbing_randomrestart,
    'Simulated-Annealing - SA': simulated_annealing,
    'GA': genetic_algorithm
    # ...
}

configs = {
    "GA":{
        "1": [1000, 20, 2, False, False], # Escolha Aleatório com Luta Determinística
        "2": [1000, 20, 2, True, False],  # Escolha Estocástica com Luta Determinística
        "3": [1000, 20, 2, False, True],  # Escolha Aleatória com Luta Estocástica
        "4": [1000, 20, 2, True, False],  # Escolha Estocástica com Luta Estocástica
    }
} 

# numero de vezes que executará cada algoritmo
n_vezes = 20
# Executa N vezes para gerar estatísticas da variável custo
# DataFrame df_custo serve para armazenar todos os resultados
df_custo, lista_custos, dict_melhores_custos= executa_n_vezes(tsp_aleatorio, algoritmos, n_vezes, **configs)
