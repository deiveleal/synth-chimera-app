# import numpy as np
# import torch
# from sklearn.decomposition import PCA  # type: ignore
# from sklearn.preprocessing import StandardScaler  # type: ignore


# def genetic_algorithm(X_num, X_img, y, fitness_fn, num_generations=50, population_size=20, mutation_rate=0.1, device="cpu"):
#     """
#     Genetic Algorithm for feature selection on multimodal data.

#     Args:
#         X_num (torch.Tensor): Structured numerical data.
#         X_img (torch.Tensor): Image data.
#         y (torch.Tensor): Labels.
#         fitness_fn (callable): Fitness function to evaluate feature subsets.
#         num_generations (int): Number of generations.
#         population_size (int): Size of the population.
#         mutation_rate (float): Probability of mutation.
#         device (str): Device to run the computation.

#     Returns:
#         np.ndarray: Binary mask of selected features.
#     """
#     # All features plus image
#     num_features = X_num.shape[1]+1

#     # Initialize population with random binary masks
#     population = np.random.randint(0, 2, (population_size, num_features))
#     test = torch.tensor(population)
#     while torch.all(test == 0):  # Recreate if all population are zero
#         print("Recreating population")
#         population = np.random.randint(0, 2, (population_size, num_features))

#     for generation in range(num_generations):
#         # Evaluate fitness of each individual
#         fitness_scores = []
#         for individual in population:
#             # Selecting all except the last column
#             selected_features = individual[:-1].astype(bool)
#             # Selecting just the last column
#             selected_images = individual[-1].astype(bool)
#             fitness = fitness_fn(
#                 X_num[:, selected_features], X_img, y, selected_images)
#             fitness_scores.append(fitness)

#         fitness_scores = np.array(fitness_scores)
#         print(f"Generation {generation + 1}")

#         # Selection: Retain top individuals based on fitness
#         sorted_indices = np.argsort(fitness_scores)[::-1]
#         population = population[sorted_indices[:population_size // 2]]

#         # Crossover: Combine pairs of top individuals to create offspring
#         offspring = []
#         for _ in range(population_size - len(population)):
#             parent1, parent2 = population[np.random.choice(
#                 len(population), 2, replace=False)]
#             crossover_point = np.random.randint(1, num_features - 1)
#             child = np.concatenate(
#                 (parent1[:crossover_point], parent2[crossover_point:]))
#             offspring.append(child)
#         offspring = np.array(offspring)

#         # Mutation: Randomly flip bits in the offspring
#         mutations = np.random.rand(*offspring.shape) < mutation_rate
#         offspring = np.logical_xor(offspring, mutations).astype(int)

#         # Create the new population
#         population = np.vstack((population, offspring))

#     # Return the best individual from the final generation
#     # best_individual = population[np.argmax(fitness_scores)] old
#     best_fitness_score = 0
#     best_sum = 0
#     for i, fitness in enumerate(fitness_scores):
#         if (fitness > best_fitness_score) or ((fitness == best_fitness_score) and (sum(population[i]) > best_sum)):
#             best_fitness_score = fitness
#             best_sum = sum(population[i])
#             best_individual = population[i]

#     return best_individual


# def particle_swarm_optimization(X_num, X_img, y, fitness_fn, num_particles=20, num_iterations=50, w=0.5, c1=0.5, c2=1, device="cpu"):
#     """
#     Particle Swarm Optimization for feature selection on multimodal data.

#     Args:
#         X_num (torch.Tensor): Structured numerical data.
#         X_img (torch.Tensor): Image data.
#         y (torch.Tensor): Labels.
#         fitness_fn (callable): Fitness function to evaluate feature subsets.
#         num_particles (int): Number of particles in the swarm.
#         num_iterations (int): Number of iterations.
#         w (float): Inertia weight.
#         c1 (float): Cognitive coefficient.
#         c2 (float): Social coefficient.
#         device (str): Device to run the computation.

#     Returns:
#         np.ndarray: Binary mask of selected features.
#     """
#     # Adding image feature
#     num_features = X_num.shape[1]+1

#     # Initialize particles randomly
#     particles = np.random.rand(num_particles, num_features) > 0.5
#     test = torch.tensor(particles)
#     while torch.all(test == 0):  # Recreate if all particles are zero
#         print("Recreating particles")
#         particles = np.random.rand(num_particles, num_features) > 0.5

#     velocities = np.random.rand(num_particles, num_features) * 0.1

#     # Initialize personal and global bests
#     personal_best_positions = particles.copy()
#     personal_best_scores = np.array([fitness_fn(
#         X_num[:, p[:-1].astype(bool)], X_img, y, p[-1].astype(bool)) for p in particles])
#     personal_best_sum = np.array([sum(p) for p in particles])

#     global_best_position = 0
#     global_best_score = 0
#     global_best_sum = 0
#     for i, personal_best_position in enumerate(personal_best_positions):
#         if (personal_best_scores[i] > global_best_score) or ((personal_best_scores[i] == global_best_score) and (sum(particles[i]) > global_best_sum)):
#             global_best_position = personal_best_position
#             global_best_score = personal_best_scores[i]
#             global_best_sum = sum(particles[i])

#     for iteration in range(num_iterations):
#         for i, particle in enumerate(particles):
#             # Evaluate fitness
#             selected_features = particle[:-1].astype(bool)
#             selected_images = particle[-1].astype(bool)
#             # print(f"Testing particle {i} = {particle.astype(bool)}")
#             fitness = fitness_fn(
#                 X_num[:, selected_features], X_img, y, selected_images)

#             # Update personal best
#             if (fitness > personal_best_scores[i]) or ((fitness == personal_best_scores[i]) and (sum(particle) > personal_best_sum[i])):
#                 personal_best_positions[i] = particle
#                 personal_best_scores[i] = fitness
#                 personal_best_sum[i] = sum(particle)

#             # Update global best
#             if (fitness > global_best_score) or ((fitness == global_best_score) and (sum(particle) > global_best_sum)):
#                 global_best_position = particle
#                 global_best_score = fitness
#                 global_best_sum = sum(particle)

#         # Update velocities and positions
#         for i in range(num_particles):
#             r1, r2 = np.random.rand(num_features), np.random.rand(num_features)
#             velocities[i] = (
#                 w * velocities[i]
#                 + c1 * r1 *
#                 (personal_best_positions[i].astype(
#                     int) - particles[i].astype(int))
#                 + c2 * r2 * (global_best_position.astype(int) -
#                              particles[i].astype(int))
#             )

#             sigmoid = 1 / (1 + np.exp(-velocities[i]))
#             mutation_mask = (sigmoid > 0.5).astype(int)
#             particles[i] = np.bitwise_xor(
#                 particles[i].astype(int), (mutation_mask))

#         print(f"Iteration {iteration + 1}")

#     return global_best_position


# def pca_feature_selection(X_num, n_components):
#     """
#     Perform PCA for feature selection.

#     Args:
#         X_num (np.ndarray): Numerical features.
#         n_components (int): Number of principal components to keep.

#     Returns:
#         torch.Tensor: Transformed features.
#     """
#     pca = PCA(n_components=n_components)
#     X_pca = pca.fit_transform(X_num.cpu().numpy())
#     return torch.tensor(X_pca, dtype=torch.float32), pca

# import numpy as np
# import time
# import torch # Adicionado para type hinting e operações se necessário

# def genetic_algorithm(
#     X_num_tensor: torch.Tensor,
#     X_img_tensor: torch.Tensor, # Este é o tensor de imagem completo
#     y_tensor: torch.Tensor,
#     fitness_fn, 
#     n_features: int,
#     n_population: int,
#     n_generations: int,
#     mutation_rate: float,
#     crossover_rate: float,
#     use_image_data: bool, # Esta flag indica se X_img_tensor deve ser considerado
#     device: str
# ):
#     """
#     Executa o Algoritmo Genético para seleção de features.
#     fitness_fn: uma função que aceita (individual_mask, X_num_original, X_img_tensor, y_tensor, use_image, model_architecture, device, epochs)
#                 e retorna um valor de fitness (menor é melhor).
#     """
#     print(f"DEBUG GA: Iniciando. n_generations = {n_generations}, n_population = {n_population}, n_features = {n_features}")

#     population = np.random.randint(0, 2, size=(n_population, n_features))
#     best_solution_overall = np.zeros(n_features, dtype=int)
#     best_fitness_overall = float('inf')

#     # Histórico
#     history_best_fitness = []
#     history_avg_fitness = []
#     history_features_count = []
#     history_time_per_generation = []

#     # Avaliação inicial (Geração 0)
#     start_time_gen0 = time.time()
#     initial_fitness_values = []
#     current_best_gen0_fitness = float('inf')
#     current_best_gen0_solution = np.zeros(n_features, dtype=int)

#     for i in range(n_population):
#         individual = population[i]
#         fitness = fitness_fn( # Chamada para evaluate_features
#             individual_mask=individual,
#             X_num_original=X_num_tensor,
#             X_img_tensor=X_img_tensor if use_image_data else None, # Passa X_img_tensor apenas se use_image_data for True
#             y_tensor=y_tensor,
#             # model_architecture="SimpleCNN", # Removido se evaluate_features não usa mais
#             device=device,
#             epochs=5 
#             # REMOVER use_image=use_image_data, # <--- REMOVER ESTA LINHA
#         )
#         initial_fitness_values.append(fitness)
#         if fitness < current_best_gen0_fitness:
#             current_best_gen0_fitness = fitness
#             current_best_gen0_solution = individual.copy()
    
#     best_fitness_overall = current_best_gen0_fitness
#     best_solution_overall = current_best_gen0_solution.copy()
    
#     history_best_fitness.append(best_fitness_overall)
#     history_avg_fitness.append(np.mean(initial_fitness_values) if initial_fitness_values else float('inf'))
#     history_features_count.append(np.sum(best_solution_overall))
#     history_time_per_generation.append(time.time() - start_time_gen0)
#     print(f"DEBUG GA: Gen 0 - BestFit: {best_fitness_overall:.4f}, AvgFit: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}, Time: {history_time_per_generation[-1]:.2f}s")

#     for gen in range(n_generations): # Loop principal de 0 a n_generations-1
#         start_time_gen = time.time()
#         current_generation_fitness_values = []
        
#         for i in range(n_population):
#             individual = population[i]
#             fitness = fitness_fn( # Chamada para evaluate_features
#                 individual_mask=individual,
#                 X_num_original=X_num_tensor,
#                 X_img_tensor=X_img_tensor if use_image_data else None, # Passa X_img_tensor apenas se use_image_data for True
#                 y_tensor=y_tensor,
#                 # model_architecture="SimpleCNN", # Removido se evaluate_features não usa mais
#                 device=device,
#                 epochs=5
#                 # REMOVER use_image=use_image_data, # <--- REMOVER ESTA LINHA
#             )
#             current_generation_fitness_values.append(fitness)

#         # Atualizar melhor solução geral da população atual
#         min_fitness_idx_current_gen = np.argmin(current_generation_fitness_values)
#         if current_generation_fitness_values[min_fitness_idx_current_gen] < best_fitness_overall:
#             best_fitness_overall = current_generation_fitness_values[min_fitness_idx_current_gen]
#             best_solution_overall = population[min_fitness_idx_current_gen].copy()

#         # Seleção (Exemplo: Roleta)
#         # Fitness menor é melhor, então precisamos inverter para seleção por roleta
#         # Adicionar um pequeno epsilon para evitar divisão por zero se fitness for 0
#         # E lidar com fitness infinito
#         adjusted_fitness = np.array(current_generation_fitness_values)
#         max_finite_fitness = np.max(adjusted_fitness[np.isfinite(adjusted_fitness)]) if np.any(np.isfinite(adjusted_fitness)) else 1.0
        
#         inverted_fitness_scores = []
#         for f_val in adjusted_fitness:
#             if np.isinf(f_val):
#                 inverted_fitness_scores.append(0) # Fitness infinito tem chance zero
#             else:
#                 # Inverter: maior valor para menor fitness. Adicionar 1 para evitar problemas com fitness negativo se existir.
#                 inverted_fitness_scores.append(max_finite_fitness - f_val + 1e-6) 
        
#         total_inverted_fitness = sum(inverted_fitness_scores)
#         if total_inverted_fitness == 0 or np.isnan(total_inverted_fitness): # Se todos os fitness forem inf ou inválidos
#             selection_probs = np.ones(n_population) / n_population # Seleção aleatória uniforme
#         else:
#             selection_probs = np.array([s / total_inverted_fitness for s in inverted_fitness_scores])
        
#         try:
#             selected_indices = np.random.choice(n_population, size=n_population, p=selection_probs)
#             new_population = population[selected_indices].copy()
#         except ValueError as e: # Se as probabilidades não somarem 1
#             print(f"DEBUG GA: Erro na seleção por roleta (probs: {selection_probs}, sum: {np.sum(selection_probs)}): {e}. Usando seleção aleatória.")
#             selected_indices = np.random.choice(n_population, size=n_population) # Fallback
#             new_population = population[selected_indices].copy()

#         # Crossover
#         for i in range(0, n_population - 1, 2): # Garante pares
#             if np.random.rand() < crossover_rate:
#                 parent1, parent2 = new_population[i], new_population[i+1]
#                 if n_features > 1:
#                     crossover_point = np.random.randint(1, n_features) # Ponto entre 1 e n_features-1
#                     child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
#                     child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
#                     new_population[i], new_population[i+1] = child1, child2
        
#         # Mutação
#         for i in range(n_population):
#             for j in range(n_features):
#                 if np.random.rand() < mutation_rate:
#                     new_population[i, j] = 1 - new_population[i, j] # Flip bit
        
#         population = new_population
        
#         end_time_gen = time.time()
#         history_time_per_generation.append(end_time_gen - start_time_gen)
#         history_best_fitness.append(best_fitness_overall) # Melhor fitness encontrado ATÉ esta geração
#         history_avg_fitness.append(np.mean(current_generation_fitness_values) if current_generation_fitness_values else float('inf'))
#         history_features_count.append(np.sum(best_solution_overall)) # Features da melhor solução ATÉ esta geração
        
#         print(f"DEBUG GA: Gen {gen+1}/{n_generations} - BestFit: {best_fitness_overall:.4f}, AvgFit: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}, HistLen: {len(history_best_fitness)}, Time: {history_time_per_generation[-1]:.2f}s")

#     selected_features_indices = np.where(best_solution_overall == 1)[0]
#     print(f"DEBUG GA: Finalizado. Comprimento final do histórico de fitness: {len(history_best_fitness)}")
#     return {
#         'best_solution_vector': best_solution_overall,
#         'best_fitness': best_fitness_overall,
#         'selected_features_indices': selected_features_indices.tolist(), # Converter para lista para JSON/session_state
#         'selected_features_count': int(np.sum(best_solution_overall)),
#         'history': {
#             'fitness': history_best_fitness,
#             'avg_fitness': history_avg_fitness,
#             'features_count': history_features_count,
#             'time_per_generation': history_time_per_generation
#         }
#     }

# def particle_swarm_optimization(
#     X_num_tensor: torch.Tensor,
#     X_img_tensor: torch.Tensor, # Este é o tensor de imagem completo
#     y_tensor: torch.Tensor,
#     fitness_fn, 
#     n_features: int,
#     n_particles: int,
#     n_iterations: int,
#     w: float, 
#     c1: float, 
#     c2: float, 
#     use_image_data: bool, # Esta flag indica se X_img_tensor deve ser considerado
#     device: str
# ):
#     print(f"DEBUG PSO: Iniciando. n_iterations = {n_iterations}, n_particles = {n_particles}, n_features = {n_features}")
    
#     # Inicialização das partículas
#     particles_position = np.random.randint(0, 2, size=(n_particles, n_features)) # Posições binárias
#     particles_velocity = np.random.uniform(-1, 1, size=(n_particles, n_features)) # Velocidades contínuas

#     pbest_position = particles_position.copy()
#     pbest_value = np.full(n_particles, float('inf'))
    
#     gbest_position = np.zeros(n_features, dtype=int)
#     gbest_value = float('inf')

#     # Histórico
#     history_gbest_fitness = []
#     history_avg_pbest_fitness = [] # Média dos pbest_values
#     history_gbest_features_count = [] # Do gbest
#     history_time_per_iteration = []

#     # Avaliação inicial (Iteração 0)
#     start_time_iter0 = time.time()
#     current_iter0_fitness_values = []
#     for i in range(n_particles):
#         fitness = fitness_fn( # Chamada para evaluate_features
#             individual_mask=particles_position[i],
#             X_num_original=X_num_tensor,
#             X_img_tensor=X_img_tensor if use_image_data else None, # Passa X_img_tensor apenas se use_image_data for True
#             y_tensor=y_tensor,
#             # model_architecture="SimpleCNN", # Removido se evaluate_features não usa mais
#             device=device,
#             epochs=5
#             # REMOVER use_image=use_image_data, # <--- REMOVER ESTA LINHA
#         )
#         pbest_value[i] = fitness
#         current_iter0_fitness_values.append(fitness) # Apenas para média inicial
#         if fitness < gbest_value:
#             gbest_value = fitness
#             gbest_position = particles_position[i].copy()

#     history_gbest_fitness.append(gbest_value)
#     history_avg_pbest_fitness.append(np.mean(pbest_value) if pbest_value.size > 0 else float('inf'))
#     history_gbest_features_count.append(np.sum(gbest_position))
#     history_time_per_iteration.append(time.time() - start_time_iter0)
#     print(f"DEBUG PSO: Iter 0 - GBestFit: {gbest_value:.4f}, AvgPBestFit: {history_avg_pbest_fitness[-1]:.4f}, GBestFeatures: {history_gbest_features_count[-1]}, Time: {history_time_per_iteration[-1]:.2f}s")

#     for iteration in range(n_iterations): # Loop principal de 0 a n_iterations-1
#         start_time_iter = time.time()
#         current_iteration_fitness_values = [] # Fitness das partículas na iteração atual

#         for i in range(n_particles):
#             fitness = fitness_fn( # Chamada para evaluate_features
#                 individual_mask=particles_position[i],
#                 X_num_original=X_num_tensor,
#                 X_img_tensor=X_img_tensor if use_image_data else None, # Passa X_img_tensor apenas se use_image_data for True
#                 y_tensor=y_tensor,
#                 # model_architecture="SimpleCNN", # Removido se evaluate_features não usa mais
#                 device=device,
#                 epochs=5
#                 # REMOVER use_image=use_image_data, # <--- REMOVER ESTA LINHA
#             )
#             current_iteration_fitness_values.append(fitness)

#             # Atualizar pbest
#             if fitness < pbest_value[i]:
#                 pbest_value[i] = fitness
#                 pbest_position[i] = particles_position[i].copy()

#             # Atualizar gbest
#             if fitness < gbest_value:
#                 gbest_value = fitness
#                 gbest_position = particles_position[i].copy()
        
#         # Atualizar velocidade e posição
#         for i in range(n_particles):
#             r1, r2 = np.random.rand(), np.random.rand()
#             inertia_term = w * particles_velocity[i]
#             cognitive_term = c1 * r1 * (pbest_position[i] - particles_position[i])
#             social_term = c2 * r2 * (gbest_position - particles_position[i])
            
#             particles_velocity[i] = inertia_term + cognitive_term + social_term
            
#             # Limitar velocidade (opcional, mas pode ajudar)
#             # particles_velocity[i] = np.clip(particles_velocity[i], -V_MAX, V_MAX) 
            
#             # Aplicar sigmoide para converter velocidade em probabilidade de ser 1 (PSO Binário)
#             prob_of_one = 1 / (1 + np.exp(-particles_velocity[i]))
#             particles_position[i] = (np.random.rand(n_features) < prob_of_one).astype(int)

#         end_time_iter = time.time()
#         history_time_per_iteration.append(end_time_iter - start_time_iter)
#         history_gbest_fitness.append(gbest_value) # Melhor fitness global encontrado ATÉ esta iteração
#         history_avg_pbest_fitness.append(np.mean(pbest_value) if pbest_value.size > 0 else float('inf')) # Média dos pbest
#         history_gbest_features_count.append(np.sum(gbest_position)) # Features da melhor solução global ATÉ esta iteração

#         print(f"DEBUG PSO: Iter {iteration+1}/{n_iterations} - GBestFit: {gbest_value:.4f}, AvgPBestFit: {history_avg_pbest_fitness[-1]:.4f}, GBestFeatures: {history_gbest_features_count[-1]}, HistLen: {len(history_gbest_fitness)}, Time: {history_time_per_iteration[-1]:.2f}s")

#     selected_features_indices = np.where(gbest_position == 1)[0]
#     print(f"DEBUG PSO: Finalizado. Comprimento final do histórico de GBest fitness: {len(history_gbest_fitness)}")
#     return {
#         'best_solution_vector': gbest_position,
#         'best_fitness': gbest_value,
#         'selected_features_indices': selected_features_indices.tolist(),
#         'selected_features_count': int(np.sum(gbest_position)),
#         'history': {
#             'fitness': history_gbest_fitness, # Renomeado para 'fitness' para consistência com GA e visualização
#             'avg_fitness': history_avg_pbest_fitness, # Renomeado para 'avg_fitness'
#             'features_count': history_gbest_features_count, # Renomeado para 'features_count'
#             'time_per_iteration': history_time_per_iteration
#         }
#     }

# import numpy as np
# import torch
# import time

# # --- GENETIC ALGORITHM ---
# def genetic_algorithm(
#     X_num_tensor: torch.Tensor, X_img_tensor: torch.Tensor, y_tensor: torch.Tensor,
#     fitness_fn, n_features: int, n_population: int, n_generations: int,
#     mutation_rate: float, crossover_rate: float, use_image_data: bool, device: str
# ):
#     population = np.random.randint(0, 2, size=(n_population, n_features), dtype=bool) if n_features > 0 else np.empty((n_population, 0), dtype=bool)
    
#     best_solution_overall = None
#     # INICIALIZAR PARA MAXIMIZAÇÃO
#     best_fitness_overall = -float('inf') 

#     history_best_fitness_overall = []
#     history_gen_best_fitness = []
#     history_avg_fitness = []
#     history_features_count = []
#     history_time_per_generation = []

#     # Avaliação inicial
#     initial_fitness_values = []
#     current_gen_best_fitness_val = -float('inf')

#     for i in range(n_population):
#         individual = population[i]
#         fitness = fitness_fn(
#             individual_mask=individual if n_features > 0 else None,
#             X_num_original=X_num_tensor if n_features > 0 else None,
#             X_img_tensor=X_img_tensor if use_image_data else None,
#             y_tensor=y_tensor, device=device, epochs=5
#         )
#         initial_fitness_values.append(fitness)
#         # MAXIMIZAÇÃO
#         if fitness > current_gen_best_fitness_val:
#             current_gen_best_fitness_val = fitness
#         if fitness > best_fitness_overall:
#             best_fitness_overall = fitness
#             best_solution_overall = individual.copy()
#         # Se fitness é igual, preferir a solução com menos features
#         elif fitness == best_fitness_overall and best_solution_overall is not None and n_features > 0:
#             if np.sum(individual) < np.sum(best_solution_overall):
#                 best_solution_overall = individual.copy()


#     history_best_fitness_overall.append(best_fitness_overall if best_fitness_overall != -float('inf') else 0.0)
#     history_gen_best_fitness.append(current_gen_best_fitness_val if current_gen_best_fitness_val != -float('inf') else 0.0)
#     history_avg_fitness.append(np.mean(initial_fitness_values) if initial_fitness_values else 0.0)
#     history_features_count.append(np.sum(best_solution_overall) if best_solution_overall is not None else 0)
#     history_time_per_generation.append(0)

#     for gen in range(n_generations):
#         start_time_gen = time.time()
#         fitness_values_this_gen = []
#         current_gen_best_fitness_val = -float('inf')

#         for i in range(n_population):
#             individual = population[i]
#             fitness = fitness_fn(
#                 individual_mask=individual if n_features > 0 else None,
#                 X_num_original=X_num_tensor if n_features > 0 else None,
#                 X_img_tensor=X_img_tensor if use_image_data else None,
#                 y_tensor=y_tensor, device=device, epochs=5
#             )
#             fitness_values_this_gen.append(fitness)
#             # MAXIMIZAÇÃO
#             if fitness > current_gen_best_fitness_val:
#                 current_gen_best_fitness_val = fitness
#             if fitness > best_fitness_overall:
#                 best_fitness_overall = fitness
#                 best_solution_overall = individual.copy()
#             # Se fitness é igual, preferir a solução com menos features
#             elif fitness == best_fitness_overall and best_solution_overall is not None and n_features > 0:
#                  if np.sum(individual) < np.sum(best_solution_overall):
#                     best_solution_overall = individual.copy()
        
#         # Seleção por Roleta (ajustada para maximização, fitness >= 0)
#         # Fitnesses negativos ou muito pequenos podem ser problemáticos para roleta.
#         # Assumindo que fitness (acurácia) é >= 0.
#         # Se o fitness pode ser 0, somar um pequeno epsilon para evitar divisão por zero se todos forem 0.
#         fitness_for_selection = [f if f > 0 else 0 for f in fitness_values_this_gen] # Tratar fitness 0 ou negativo
#         total_fitness_for_selection = sum(fitness_for_selection)

#         if total_fitness_for_selection == 0 or not fitness_values_this_gen:
#             # Seleção uniforme se todos os fitness são 0 ou lista vazia
#             selection_probs = [1.0/n_population] * n_population
#         else:
#             selection_probs = [f / total_fitness_for_selection for f in fitness_for_selection]
        
#         if np.isnan(selection_probs).any() or not np.isclose(sum(selection_probs), 1.0):
#             # Fallback para seleção uniforme se as probabilidades estiverem ruins
#             # print(f"Warning GA Gen {gen+1}: Problema com probabilidades de seleção ({selection_probs}), usando uniforme.")
#             selection_probs = [1.0/n_population] * n_population

#         parent_indices = np.random.choice(n_population, size=n_population, p=selection_probs)
#         selected_parents = population[parent_indices]

#         # Crossover e Mutação (lógica permanece a mesma)
#         new_population = np.empty_like(population)
#         for i in range(0, n_population, 2):
#             if i + 1 < n_population and n_features > 0:
#                 parent1, parent2 = selected_parents[i], selected_parents[i+1]
#                 if np.random.rand() < crossover_rate:
#                     point = np.random.randint(1, n_features -1 if n_features > 1 else 1)
#                     child1 = np.concatenate((parent1[:point], parent2[point:]))
#                     child2 = np.concatenate((parent2[:point], parent1[point:]))
#                 else:
#                     child1, child2 = parent1.copy(), parent2.copy()
#                 for child in [child1, child2]:
#                     for j in range(n_features):
#                         if np.random.rand() < mutation_rate: child[j] = not child[j]
#                 new_population[i] = child1
#                 if i + 1 < n_population: new_population[i+1] = child2
#             elif n_features > 0 :
#                 new_population[i] = selected_parents[i].copy()
#                 for j in range(n_features):
#                     if np.random.rand() < mutation_rate: new_population[i][j] = not new_population[i][j]
#             elif n_features == 0:
#                  new_population[i] = np.array([], dtype=bool)
#                  if i + 1 < n_population: new_population[i+1] = np.array([], dtype=bool)
#         population = new_population
        
#         end_time_gen = time.time()
#         history_time_per_generation.append(int(end_time_gen - start_time_gen))
#         history_best_fitness_overall.append(best_fitness_overall if best_fitness_overall != -float('inf') else 0.0)
#         history_gen_best_fitness.append(current_gen_best_fitness_val if current_gen_best_fitness_val != -float('inf') else 0.0)
#         history_avg_fitness.append(np.mean(fitness_values_this_gen) if fitness_values_this_gen else 0.0)
#         history_features_count.append(np.sum(best_solution_overall) if best_solution_overall is not None else 0)

#     final_selected_indices = np.where(best_solution_overall == 1)[0].tolist() if best_solution_overall is not None and n_features > 0 else []
#     return {
#         'best_solution_vector': best_solution_overall.tolist() if best_solution_overall is not None else ([False]*n_features if n_features > 0 else []),
#         'best_fitness': best_fitness_overall if best_fitness_overall != -float('inf') else 0.0,
#         'selected_features_indices': final_selected_indices,
#         'selected_features_count': len(final_selected_indices),
#         'history': {
#             'fitness_overall': history_best_fitness_overall,
#             'fitness_epoch': history_gen_best_fitness,
#             'avg_fitness_epoch': history_avg_fitness,
#             'features_count': history_features_count,
#             'time_per_epoch': history_time_per_generation
#         }
#     }

# # --- PARTICLE SWARM OPTIMIZATION ---
# def particle_swarm_optimization(
#     X_num_tensor: torch.Tensor, X_img_tensor: torch.Tensor, y_tensor: torch.Tensor,
#     fitness_fn, n_features: int, n_particles: int, n_iterations: int,
#     w: float, c1: float, c2: float, use_image_data: bool, device: str
# ):
#     particles_position = np.random.rand(n_particles, n_features) > 0.5 if n_features > 0 else np.empty((n_particles, 0), dtype=bool)
#     particles_velocity = np.random.rand(n_particles, n_features) * 0.1 if n_features > 0 else np.empty((n_particles, 0), dtype=float)

#     # INICIALIZAR PARA MAXIMIZAÇÃO
#     pbest_value = np.full(n_particles, -float('inf'))
#     pbest_position = particles_position.copy() # ou inicializar com None e copiar na primeira avaliação
    
#     gbest_position = None
#     gbest_value = -float('inf')

#     history_best_fitness_overall = []
#     history_iter_best_fitness = []
#     history_avg_fitness = []
#     history_features_count = []
#     history_time_per_iteration = []

#     # Avaliação inicial
#     initial_fitness_values = []
#     current_iter_best_fitness_val = -float('inf')

#     for i in range(n_particles):
#         fitness = fitness_fn(
#             individual_mask=particles_position[i] if n_features > 0 else None,
#             X_num_original=X_num_tensor if n_features > 0 else None,
#             X_img_tensor=X_img_tensor if use_image_data else None,
#             y_tensor=y_tensor, device=device, epochs=5
#         )
#         initial_fitness_values.append(fitness)
#         pbest_value[i] = fitness # pbest inicial
#         pbest_position[i] = particles_position[i].copy()

#         # MAXIMIZAÇÃO
#         if fitness > current_iter_best_fitness_val:
#             current_iter_best_fitness_val = fitness
#         if fitness > gbest_value:
#             gbest_value = fitness
#             gbest_position = particles_position[i].copy()
#         # Se fitness é igual, preferir a solução com menos features
#         elif fitness == gbest_value and gbest_position is not None and n_features > 0:
#             if np.sum(particles_position[i]) < np.sum(gbest_position):
#                 gbest_position = particles_position[i].copy()

#     history_best_fitness_overall.append(gbest_value if gbest_value != -float('inf') else 0.0)
#     history_iter_best_fitness.append(current_iter_best_fitness_val if current_iter_best_fitness_val != -float('inf') else 0.0)
#     history_avg_fitness.append(np.mean(initial_fitness_values) if initial_fitness_values else 0.0)
#     history_features_count.append(np.sum(gbest_position) if gbest_position is not None else 0)
#     history_time_per_iteration.append(0)

#     for iteration in range(n_iterations):
#         start_time_iter = time.time()
#         fitness_values_this_iter = []
#         current_iter_best_fitness_val = -float('inf')

#         for i in range(n_particles):
#             if n_features > 0 and gbest_position is not None: # gbest_position deve existir após a avaliação inicial
#                 r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
#                 cognitive_velocity = c1 * r1 * (pbest_position[i].astype(float) - particles_position[i].astype(float))
#                 social_velocity = c2 * r2 * (gbest_position.astype(float) - particles_position[i].astype(float))
#                 particles_velocity[i] = w * particles_velocity[i] + cognitive_velocity + social_velocity
                
#                 sigmoid_val = 1 / (1 + np.exp(-particles_velocity[i]))
#                 particles_position[i] = np.random.rand(n_features) < sigmoid_val
            
#             fitness = fitness_fn(
#                 individual_mask=particles_position[i] if n_features > 0 else None,
#                 X_num_original=X_num_tensor if n_features > 0 else None,
#                 X_img_tensor=X_img_tensor if use_image_data else None,
#                 y_tensor=y_tensor, device=device, epochs=5
#             )
#             fitness_values_this_iter.append(fitness)

#             # MAXIMIZAÇÃO
#             if fitness > current_iter_best_fitness_val:
#                  current_iter_best_fitness_val = fitness
#             if fitness > pbest_value[i]:
#                 pbest_value[i] = fitness
#                 pbest_position[i] = particles_position[i].copy()
#             if fitness > gbest_value:
#                 gbest_value = fitness
#                 gbest_position = particles_position[i].copy()
#             # Se fitness é igual, preferir a solução com menos features
#             elif fitness == gbest_value and gbest_position is not None and n_features > 0:
#                 if np.sum(particles_position[i]) < np.sum(gbest_position):
#                     gbest_position = particles_position[i].copy()

#         end_time_iter = time.time()
#         history_time_per_iteration.append(int(end_time_iter - start_time_iter))
#         history_best_fitness_overall.append(gbest_value if gbest_value != -float('inf') else 0.0)
#         history_iter_best_fitness.append(current_iter_best_fitness_val if current_iter_best_fitness_val != -float('inf') else 0.0)
#         history_avg_fitness.append(np.mean(fitness_values_this_iter) if fitness_values_this_iter else 0.0)
#         history_features_count.append(np.sum(gbest_position) if gbest_position is not None else 0)

#     final_selected_indices = np.where(gbest_position == 1)[0].tolist() if gbest_position is not None and n_features > 0 else []
#     return {
#         'best_solution_vector': gbest_position.tolist() if gbest_position is not None else ([False]*n_features if n_features > 0 else []),
#         'best_fitness': gbest_value if gbest_value != -float('inf') else 0.0,
#         'selected_features_indices': final_selected_indices,
#         'selected_features_count': len(final_selected_indices),
#         'history': {
#             'fitness_overall': history_best_fitness_overall,
#             'fitness_epoch': history_iter_best_fitness,
#             'avg_fitness_epoch': history_avg_fitness,
#             'features_count': history_features_count,
#             'time_per_epoch': history_time_per_iteration
#         }
#     }

import numpy as np
import torch
import time

# --- GENETIC ALGORITHM ---
def genetic_algorithm(
    X_num_tensor: torch.Tensor, X_img_tensor: torch.Tensor, y_tensor: torch.Tensor,
    fitness_fn, n_features: int, n_population: int, n_generations: int,
    mutation_rate: float, crossover_rate: float, use_image_data: bool, device: str
):
    print(f"\nDEBUG GA: Starting Genetic Algorithm. n_features={n_features}, n_population={n_population}, n_generations={n_generations}")
    population = np.random.randint(0, 2, size=(n_population, n_features), dtype=bool) if n_features > 0 else np.empty((n_population, 0), dtype=bool)
    if n_features > 0: print(f"DEBUG GA: Initial population (first 3): \n{population[:3]}")
    
    best_solution_overall = None
    best_fitness_overall = -float('inf') 

    history_best_fitness_overall = []
    history_gen_best_fitness = []
    history_avg_fitness = []
    history_features_count = []
    history_time_per_generation = []

    print("DEBUG GA: Starting initial evaluation...")
    initial_fitness_values = []
    current_gen_best_fitness_val = -float('inf')

    for i in range(n_population):
        individual = population[i]
        print(f"DEBUG GA Initial Eval: Evaluating individual {i}")
        fitness = fitness_fn(
            individual_mask=individual if n_features > 0 else None,
            X_num_original=X_num_tensor if n_features > 0 else None,
            X_img_tensor=X_img_tensor if use_image_data else None,
            y_tensor=y_tensor, device=device, epochs=5 # epochs para fitness_fn
        )
        print(f"DEBUG GA Initial Eval: Individual {i}, Fitness: {fitness:.4f}, Mask: {individual.astype(int) if n_features > 0 else 'N/A'}")
        initial_fitness_values.append(fitness)
        if fitness > current_gen_best_fitness_val:
            current_gen_best_fitness_val = fitness
        if fitness > best_fitness_overall:
            best_fitness_overall = fitness
            best_solution_overall = individual.copy()
            print(f"DEBUG GA Initial Eval: New best_overall_fitness: {best_fitness_overall:.4f}, Solution: {best_solution_overall.astype(int) if n_features > 0 else 'N/A'}")
        elif fitness == best_fitness_overall and best_solution_overall is not None and n_features > 0:
            if np.sum(individual) < np.sum(best_solution_overall):
                best_solution_overall = individual.copy()
                print(f"DEBUG GA Initial Eval: Same best_overall_fitness {best_fitness_overall:.4f} but fewer features. Solution: {best_solution_overall.astype(int)}")

    history_best_fitness_overall.append(best_fitness_overall if best_fitness_overall != -float('inf') else 0.0)
    history_gen_best_fitness.append(current_gen_best_fitness_val if current_gen_best_fitness_val != -float('inf') else 0.0)
    history_avg_fitness.append(np.mean(initial_fitness_values) if initial_fitness_values else 0.0)
    history_features_count.append(np.sum(best_solution_overall) if best_solution_overall is not None else 0)
    history_time_per_generation.append(0)
    print(f"DEBUG GA: After Initial Eval - BestOverall: {history_best_fitness_overall[-1]:.4f}, GenBest: {history_gen_best_fitness[-1]:.4f}, GenAvg: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}")

    for gen in range(n_generations):
        print(f"\nDEBUG GA: Starting Generation {gen + 1}/{n_generations}")
        start_time_gen = time.time()
        fitness_values_this_gen = []
        current_gen_best_fitness_val = -float('inf')

        for i in range(n_population):
            individual = population[i]
            print(f"DEBUG GA Gen {gen+1}: Evaluating individual {i}")
            fitness = fitness_fn(
                individual_mask=individual if n_features > 0 else None,
                X_num_original=X_num_tensor if n_features > 0 else None,
                X_img_tensor=X_img_tensor if use_image_data else None,
                y_tensor=y_tensor, device=device, epochs=5
            )
            print(f"DEBUG GA Gen {gen+1}: Individual {i}, Fitness: {fitness:.4f}, Mask: {individual.astype(int) if n_features > 0 else 'N/A'}")
            fitness_values_this_gen.append(fitness)
            if fitness > current_gen_best_fitness_val:
                current_gen_best_fitness_val = fitness
            if fitness > best_fitness_overall:
                best_fitness_overall = fitness
                best_solution_overall = individual.copy()
                print(f"DEBUG GA Gen {gen+1}: New best_overall_fitness: {best_fitness_overall:.4f}, Solution: {best_solution_overall.astype(int) if n_features > 0 else 'N/A'}")
            elif fitness == best_fitness_overall and best_solution_overall is not None and n_features > 0:
                 if np.sum(individual) < np.sum(best_solution_overall):
                    best_solution_overall = individual.copy()
                    print(f"DEBUG GA Gen {gen+1}: Same best_overall_fitness {best_fitness_overall:.4f} but fewer features. Solution: {best_solution_overall.astype(int)}")
        
        fitness_for_selection = [f if f > 0 else 0 for f in fitness_values_this_gen]
        total_fitness_for_selection = sum(fitness_for_selection)

        if total_fitness_for_selection == 0 or not fitness_values_this_gen:
            selection_probs = [1.0/n_population] * n_population
            print(f"DEBUG GA Gen {gen+1}: Using uniform selection probabilities (total_fitness_for_selection={total_fitness_for_selection}).")
        else:
            selection_probs = [f / total_fitness_for_selection for f in fitness_for_selection]
        
        if np.isnan(selection_probs).any() or not np.isclose(sum(selection_probs), 1.0):
            print(f"DEBUG GA Gen {gen+1}: Warning - Problem with selection_probs ({selection_probs}), using uniform.")
            selection_probs = [1.0/n_population] * n_population

        parent_indices = np.random.choice(n_population, size=n_population, p=selection_probs)
        selected_parents = population[parent_indices]
        if n_features > 0: print(f"DEBUG GA Gen {gen+1}: Selected parents (first 3): \n{selected_parents[:3]}")

        new_population = np.empty_like(population)
        for i in range(0, n_population, 2):
            if i + 1 < n_population and n_features > 0:
                parent1, parent2 = selected_parents[i], selected_parents[i+1]
                if np.random.rand() < crossover_rate:
                    point = np.random.randint(1, n_features -1 if n_features > 1 else 1)
                    child1 = np.concatenate((parent1[:point], parent2[point:]))
                    child2 = np.concatenate((parent2[:point], parent1[point:]))
                    print(f"DEBUG GA Gen {gen+1}: Crossover at point {point} for pair {i//2}")
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                for child_idx, child in enumerate([child1, child2]):
                    original_child_mask = child.copy().astype(int) if n_features > 0 else []
                    mutated = False
                    for j in range(n_features):
                        if np.random.rand() < mutation_rate:
                            child[j] = not child[j]
                            mutated = True
                    if mutated and n_features > 0:
                        print(f"DEBUG GA Gen {gen+1}: Mutation for child {child_idx} of pair {i//2}. Before: {original_child_mask}, After: {child.astype(int)}")
                new_population[i] = child1
                if i + 1 < n_population: new_population[i+1] = child2
            elif n_features > 0 :
                new_population[i] = selected_parents[i].copy()
                original_child_mask = new_population[i].copy().astype(int)
                mutated = False
                for j in range(n_features):
                    if np.random.rand() < mutation_rate:
                        new_population[i][j] = not new_population[i][j]
                        mutated = True
                if mutated:
                    print(f"DEBUG GA Gen {gen+1}: Mutation for single parent {i}. Before: {original_child_mask}, After: {new_population[i].astype(int)}")
            elif n_features == 0:
                 new_population[i] = np.array([], dtype=bool)
                 if i + 1 < n_population: new_population[i+1] = np.array([], dtype=bool)
        population = new_population
        if n_features > 0: print(f"DEBUG GA Gen {gen+1}: New population (first 3): \n{population[:3]}")
        
        end_time_gen = time.time()
        history_time_per_generation.append(int(end_time_gen - start_time_gen))
        history_best_fitness_overall.append(best_fitness_overall if best_fitness_overall != -float('inf') else 0.0)
        history_gen_best_fitness.append(current_gen_best_fitness_val if current_gen_best_fitness_val != -float('inf') else 0.0)
        history_avg_fitness.append(np.mean(fitness_values_this_gen) if fitness_values_this_gen else 0.0)
        history_features_count.append(np.sum(best_solution_overall) if best_solution_overall is not None else 0)
        print(f"DEBUG GA: Gen {gen+1}/{n_generations} - BestOverall: {history_best_fitness_overall[-1]:.4f}, GenBest: {history_gen_best_fitness[-1]:.4f}, GenAvg: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}, Time: {history_time_per_generation[-1]:.2f}s")

    final_selected_indices = np.where(best_solution_overall == 1)[0].tolist() if best_solution_overall is not None and n_features > 0 else []
    final_result = {
        'best_solution_vector': best_solution_overall.tolist() if best_solution_overall is not None else ([False]*n_features if n_features > 0 else []),
        'best_fitness': best_fitness_overall if best_fitness_overall != -float('inf') else 0.0,
        'selected_features_indices': final_selected_indices,
        'selected_features_count': len(final_selected_indices),
        'history': {
            'fitness_overall': history_best_fitness_overall,
            'fitness_epoch': history_gen_best_fitness, # Corrigido para usar a variável correta
            'avg_fitness_epoch': history_avg_fitness,
            'features_count': history_features_count,
            'time_per_epoch': history_time_per_generation
        }
    }
    print(f"DEBUG GA: Finished. Final best_fitness: {final_result['best_fitness']:.4f}, Final_solution_vector: {final_result['best_solution_vector']}")
    return final_result

# --- PARTICLE SWARM OPTIMIZATION ---
def particle_swarm_optimization(
    X_num_tensor: torch.Tensor, X_img_tensor: torch.Tensor, y_tensor: torch.Tensor,
    fitness_fn, n_features: int, n_particles: int, n_iterations: int,
    w: float, c1: float, c2: float, use_image_data: bool, device: str
):
    print(f"\nDEBUG PSO: Starting Particle Swarm Optimization. n_features={n_features}, n_particles={n_particles}, n_iterations={n_iterations}")
    particles_position = np.random.rand(n_particles, n_features) > 0.5 if n_features > 0 else np.empty((n_particles, 0), dtype=bool)
    particles_velocity = np.random.rand(n_particles, n_features) * 0.1 if n_features > 0 else np.empty((n_particles, 0), dtype=float)
    # if n_features > 0: print(f"DEBUG PSO: Initial particle positions (first 3): \n{particles_position[:3].astype(int)}")

    pbest_value = np.full(n_particles, -float('inf'))
    pbest_position = particles_position.copy() 
    
    gbest_position = None
    gbest_value = -float('inf')

    history_best_fitness_overall = []
    history_iter_best_fitness = []
    history_avg_fitness = []
    history_features_count = []
    history_time_per_iteration = []

    print("DEBUG PSO: Starting initial evaluation...")
    initial_fitness_values = []
    current_iter_best_fitness_val = -float('inf')

    for i in range(n_particles):
        print(f"DEBUG PSO Initial Eval: Evaluating particle {i}")
        fitness = fitness_fn(
            individual_mask=particles_position[i] if n_features > 0 else None,
            X_num_original=X_num_tensor if n_features > 0 else None,
            X_img_tensor=X_img_tensor if use_image_data else None,
            y_tensor=y_tensor, device=device, epochs=5
        )
        print(f"DEBUG PSO Initial Eval: Particle {i}, Fitness: {fitness:.4f}, Position: {particles_position[i].astype(int) if n_features > 0 else 'N/A'}")
        initial_fitness_values.append(fitness)
        pbest_value[i] = fitness 
        pbest_position[i] = particles_position[i].copy()

        if fitness > current_iter_best_fitness_val:
            current_iter_best_fitness_val = fitness
        if fitness > gbest_value:
            gbest_value = fitness
            gbest_position = particles_position[i].copy()
            print(f"DEBUG PSO Initial Eval: New gbest_value: {gbest_value:.4f}, Position: {gbest_position.astype(int) if n_features > 0 else 'N/A'}")
        elif fitness == gbest_value and gbest_position is not None and n_features > 0:
            if np.sum(particles_position[i]) < np.sum(gbest_position):
                gbest_position = particles_position[i].copy()
                print(f"DEBUG PSO Initial Eval: Same gbest_value {gbest_value:.4f} but fewer features. Position: {gbest_position.astype(int)}")

    history_best_fitness_overall.append(gbest_value if gbest_value != -float('inf') else 0.0)
    history_iter_best_fitness.append(current_iter_best_fitness_val if current_iter_best_fitness_val != -float('inf') else 0.0)
    history_avg_fitness.append(np.mean(initial_fitness_values) if initial_fitness_values else 0.0)
    history_features_count.append(np.sum(gbest_position) if gbest_position is not None else 0)
    history_time_per_iteration.append(0)
    print(f"DEBUG PSO: After Initial Eval - GBestOverall: {history_best_fitness_overall[-1]:.4f}, IterBest: {history_iter_best_fitness[-1]:.4f}, IterAvg: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}")

    for iteration in range(n_iterations):
        print(f"\nDEBUG PSO: Starting Iteration {iteration + 1}/{n_iterations}")
        start_time_iter = time.time()
        fitness_values_this_iter = []
        current_iter_best_fitness_val = -float('inf')

        for i in range(n_particles):
            if n_features > 0 and gbest_position is not None: 
                r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
                cognitive_velocity = c1 * r1 * (pbest_position[i].astype(float) - particles_position[i].astype(float))
                social_velocity = c2 * r2 * (gbest_position.astype(float) - particles_position[i].astype(float))
                particles_velocity[i] = w * particles_velocity[i] + cognitive_velocity + social_velocity
                
                sigmoid_val = 1 / (1 + np.exp(-particles_velocity[i])) # Element-wise sigmoid
                particles_position[i] = np.random.rand(n_features) < sigmoid_val # Binarize
                print(f"DEBUG PSO Iter {iteration+1}, Particle {i}: Velocity (sum): {np.sum(particles_velocity[i]):.2f}, Sigmoid (mean): {np.mean(sigmoid_val):.2f}, New Pos: {particles_position[i].astype(int)}")
            
            print(f"DEBUG PSO Iter {iteration+1}: Evaluating particle {i}")
            fitness = fitness_fn(
                individual_mask=particles_position[i] if n_features > 0 else None,
                X_num_original=X_num_tensor if n_features > 0 else None,
                X_img_tensor=X_img_tensor if use_image_data else None,
                y_tensor=y_tensor, device=device, epochs=5
            )
            print(f"DEBUG PSO Iter {iteration+1}: Particle {i}, Fitness: {fitness:.4f}, Position: {particles_position[i].astype(int) if n_features > 0 else 'N/A'}")
            fitness_values_this_iter.append(fitness)

            if fitness > current_iter_best_fitness_val:
                 current_iter_best_fitness_val = fitness
            if fitness > pbest_value[i]:
                pbest_value[i] = fitness
                pbest_position[i] = particles_position[i].copy()
                print(f"DEBUG PSO Iter {iteration+1}, Particle {i}: New pbest_value: {pbest_value[i]:.4f}")
            if fitness > gbest_value:
                gbest_value = fitness
                gbest_position = particles_position[i].copy()
                print(f"DEBUG PSO Iter {iteration+1}: New gbest_value: {gbest_value:.4f}, Position: {gbest_position.astype(int) if n_features > 0 else 'N/A'}")
            elif fitness == gbest_value and gbest_position is not None and n_features > 0:
                if np.sum(particles_position[i]) < np.sum(gbest_position):
                    gbest_position = particles_position[i].copy()
                    print(f"DEBUG PSO Iter {iteration+1}: Same gbest_value {gbest_value:.4f} but fewer features. Position: {gbest_position.astype(int)}")

        end_time_iter = time.time()
        history_time_per_iteration.append(int(end_time_iter - start_time_iter))
        history_best_fitness_overall.append(gbest_value if gbest_value != -float('inf') else 0.0)
        history_iter_best_fitness.append(current_iter_best_fitness_val if current_iter_best_fitness_val != -float('inf') else 0.0)
        history_avg_fitness.append(np.mean(fitness_values_this_iter) if fitness_values_this_iter else 0.0)
        history_features_count.append(np.sum(gbest_position) if gbest_position is not None else 0)
        print(f"DEBUG PSO: Iter {iteration+1}/{n_iterations} - GBestOverall: {history_best_fitness_overall[-1]:.4f}, IterBest: {history_iter_best_fitness[-1]:.4f}, IterAvg: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}, Time: {history_time_per_iteration[-1]:.2f}s")

    final_selected_indices = np.where(gbest_position == 1)[0].tolist() if gbest_position is not None and n_features > 0 else []
    final_result = {
        'best_solution_vector': gbest_position.tolist() if gbest_position is not None else ([False]*n_features if n_features > 0 else []),
        'best_fitness': gbest_value if gbest_value != -float('inf') else 0.0,
        'selected_features_indices': final_selected_indices,
        'selected_features_count': len(final_selected_indices),
        'history': {
            'fitness_overall': history_best_fitness_overall,
            'fitness_epoch': history_iter_best_fitness,
            'avg_fitness_epoch': history_avg_fitness,
            'features_count': history_features_count,
            'time_per_epoch': history_time_per_iteration
        }
    }
    print(f"DEBUG PSO: Finished. Final gbest_fitness: {final_result['best_fitness']:.4f}, Final_gbest_position: {final_result['best_solution_vector']}")
    return final_result