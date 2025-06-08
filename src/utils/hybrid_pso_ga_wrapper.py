import numpy as np
import torch

from .hybrid_pso_ga import Hybrid_PSO_GA


def hybrid_pso_ga_optimization(X_num, X_img, y, fitness_fn, num_generations=50, population_size=40,
                               crossover_prob=0.8, mutation_prob=0.1, ga_ratio=0.6,
                               pso_local_iterations=5, w=0.7, c1=1.5, c2=1.5, device="cpu"):
    """
    Wrapper para o algoritmo híbrido PSO-GA para seleção de características em dados multimodais.

    Args:
        X_num (torch.Tensor): Dados numéricos estruturados.
        X_img (torch.Tensor): Dados de imagem.
        y (torch.Tensor): Rótulos/classes.
        fitness_fn (callable): Função de fitness para avaliar subconjuntos de características.
        num_generations (int): Número de gerações.
        population_size (int): Tamanho da população.
        crossover_prob (float): Probabilidade de crossover.
        mutation_prob (float): Probabilidade de mutação.
        ga_ratio (float): Proporção da população inicializada usando GA puro.
        pso_local_iterations (int): Número de iterações PSO para refinamento local.
        w (float): Parâmetro de inércia do PSO.
        c1 (float): Coeficiente cognitivo do PSO.
        c2 (float): Coeficiente social do PSO.
        device (str): Dispositivo para executar os cálculos ("cpu" ou "cuda").

    Returns:
        np.ndarray: Máscara binária de características selecionadas.
    """
    # Converter tensores para numpy se necessário
    X_num_np = X_num.cpu().numpy() if hasattr(X_num, 'cpu') else X_num
    y_np = y.cpu().numpy() if hasattr(y, 'cpu') else y

    # Função de fitness personalizada que avalia usando a função de fitness fornecida
    def custom_fitness(solution):
        # Separando seleção para características numéricas e imagens
        selected_features = solution[:-1].astype(bool)
        use_image = solution[-1].astype(bool)

        # Converter arrays numpy de volta para tensores torch para o fitness_fn
        X_selected = torch.tensor(
            X_num_np[:, selected_features], device=device)
        return fitness_fn(X_selected, X_img, y, use_image)

    # Define nomes das características para facilitar a visualização
    feature_names = [f"Feature_{i}" for i in range(X_num.shape[1])]
    feature_names.append("Image_Features")

    # Número total de características (numéricas + 1 para imagem)
    n_features = X_num.shape[1] + 1

    # Inicializa o modelo híbrido
    hybrid_model = Hybrid_PSO_GA(
        # Adicionando coluna para imagem
        X=np.hstack([X_num_np, np.ones((X_num_np.shape[0], 1))]),
        y=y_np,
        n_individuals=population_size,
        n_features=n_features,
        ga_ratio=ga_ratio,
        pso_local_iterations=pso_local_iterations,
        max_generations=num_generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        w=w,
        c1=c1,
        c2=c2,
        min_features=1,  # Permitir pelo menos uma característica
        feature_names=feature_names
    )

    # Substituir o método fitness original pelo nosso fitness personalizado
    hybrid_model.fitness = custom_fitness

    # # Executar o algoritmo
    # _, _, best_solution = hybrid_model.run()

    # # Retornar a melhor solução como uma máscara binária
    # return best_solution
    hybrid_results = hybrid_model.run()  # Agora retorna um dicionário
    
    return hybrid_results  # Retornar o dicionário completo
