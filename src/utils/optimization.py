# coding=utf-8
import numpy as np
import torch
import time
from typing import Callable


def genetic_algorithm(
    X_num_tensor: torch.Tensor, 
    X_img_tensor: torch.Tensor, 
    y_tensor: torch.Tensor,
    X_num_val: torch.Tensor,        # <-- Adicionar
    X_img_val: torch.Tensor,        # <-- Adicionar
    y_val: torch.Tensor,            # <-- Adicionar
    fitness_fn: Callable, 
    n_features: int, 
    n_population: int, 
    n_generations: int,
    mutation_rate: float, 
    crossover_rate: float, 
    use_image_data: bool, 
    device: str,
    fitness_fn_epochs: int
):
    print(f"\nDEBUG GA: Iniciando Algoritmo Genético. n_features={n_features}, n_população={n_population}, n_gerações={n_generations}, epochs_fitness={fitness_fn_epochs}, mutation_rate={mutation_rate}, crossover_rate={crossover_rate}")
    population = np.random.randint(0, 2, size=(n_population, n_features), dtype=bool) if n_features > 0 else np.empty((n_population, 0), dtype=bool)

    if n_features > 0:
        for i in range(n_population):
            if not np.any(population[i]):
                random_feature_index = np.random.randint(0, n_features)
                population[i, random_feature_index] = True
                print(f"DEBUG GA: Indivíduo {i} era todo zero, ativada feature {random_feature_index}") # LOG
    
    best_solution_overall = None
    best_fitness_overall = -float('inf') 

    history_best_fitness_overall = []
    history_gen_best_fitness = []
    history_avg_fitness = []
    history_features_count = []
    history_time_per_generation = []

    print("DEBUG GA: Iniciando avaliação da Geração 0 (inicial)")
    start_time_gen0 = time.time()
    initial_fitness_values = []
    current_gen_best_fitness_val = -float('inf')

    for i in range(n_population):
        individual = population[i]
        fitness = fitness_fn(
            individual_mask=individual if n_features > 0 else None,
            X_num_train=X_num_tensor if n_features > 0 else None,     # <- CORRIGIDO
            X_img_train=X_img_tensor if use_image_data else None,     # <- CORRIGIDO
            y_train=y_tensor,                                         # <- CORRIGIDO
            X_num_val=X_num_val if n_features > 0 else None,         # <- NOVO
            X_img_val=X_img_val if use_image_data else None,         # <- NOVO
            y_val=y_val,                                             # <- NOVO
            device=device, 
            epochs=fitness_fn_epochs,
            algorithm_id="GA"
        )
        initial_fitness_values.append(fitness)
        if fitness > current_gen_best_fitness_val:
            current_gen_best_fitness_val = fitness
        
        if fitness > best_fitness_overall:
            best_fitness_overall = fitness
            best_solution_overall = individual.copy()
            print(f"DEBUG GA G0: Novo melhor_fitness_geral: {best_fitness_overall:.4f}, Solução: {best_solution_overall.astype(int) if n_features > 0 else 'N/A'}, Features: {np.sum(best_solution_overall) if best_solution_overall is not None else 0}")
        elif fitness == best_fitness_overall and n_features > 0:
            if best_solution_overall is None or (np.sum(individual) < np.sum(best_solution_overall)):
                best_solution_overall = individual.copy()
                print(f"DEBUG GA G0: Mesmo melhor_fitness_geral {best_fitness_overall:.4f} mas com menos features. Solução: {best_solution_overall.astype(int)}, Features: {np.sum(best_solution_overall)}") # LOG

    history_best_fitness_overall.append(best_fitness_overall if best_fitness_overall != -float('inf') else 0.0)
    history_gen_best_fitness.append(current_gen_best_fitness_val if current_gen_best_fitness_val != -float('inf') else 0.0)
    history_avg_fitness.append(np.mean(initial_fitness_values) if initial_fitness_values else 0.0)
    history_features_count.append(np.sum(best_solution_overall) if best_solution_overall is not None else 0)
    gen0_time = time.time() - start_time_gen0
    history_time_per_generation.append(gen0_time)
    print(f"DEBUG GA: Fim da Geração 0. MelhorFitnessGeral: {history_best_fitness_overall[-1]:.4f}, MelhorFitnessGen: {history_gen_best_fitness[-1]:.4f}, MédiaFitnessGen: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}, Tempo: {gen0_time:.2f}s")

    for gen in range(n_generations):
        print(f"\nDEBUG GA: Iniciando Geração {gen + 1}/{n_generations}")
        start_time_gen = time.time()
        fitness_values_this_gen = []
        current_gen_best_fitness_val = -float('inf')

        for i in range(n_population):
            individual = population[i]
            fitness = fitness_fn(
                individual_mask=individual if n_features > 0 else None,
                X_num_train=X_num_tensor if n_features > 0 else None,     # <- CORRIGIDO
                X_img_train=X_img_tensor if use_image_data else None,     # <- CORRIGIDO
                y_train=y_tensor,                                         # <- CORRIGIDO
                X_num_val=X_num_val if n_features > 0 else None,         # <- NOVO
                X_img_val=X_img_val if use_image_data else None,         # <- NOVO
                y_val=y_val,                                             # <- NOVO
                device=device, 
                epochs=fitness_fn_epochs,
                algorithm_id="GA"
            )
            fitness_values_this_gen.append(fitness)
            if fitness > current_gen_best_fitness_val:
                current_gen_best_fitness_val = fitness
            
            if fitness > best_fitness_overall:
                best_fitness_overall = fitness
                best_solution_overall = individual.copy()
                print(f"DEBUG GA Gen {gen+1}: Novo melhor_fitness_geral: {best_fitness_overall:.4f}, Solução: {best_solution_overall.astype(int) if n_features > 0 else 'N/A'}, Features: {np.sum(best_solution_overall) if best_solution_overall is not None else 0}")
            elif fitness == best_fitness_overall and n_features > 0:
                if best_solution_overall is None or (np.sum(individual) < np.sum(best_solution_overall)):
                    best_solution_overall = individual.copy()
                    print(f"DEBUG GA Gen {gen+1}: Mesmo melhor_fitness_geral {best_fitness_overall:.4f} mas com menos features. Solução: {best_solution_overall.astype(int)}, Features: {np.sum(best_solution_overall)}")
        
        # Seleção (Torneio Simples)
        parents_indices = []
        for _ in range(n_population):
            idx1, idx2 = np.random.choice(n_population, 2, replace=False)
            parents_indices.append(idx1 if fitness_values_this_gen[idx1] >= fitness_values_this_gen[idx2] else idx2)
        parents = population[parents_indices]
        
        # Crossover e Mutação
        offspring_population = np.empty_like(population)
        for i in range(0, n_population, 2):
            parent1, parent2 = parents[i], parents[i+1 if i+1 < n_population else i] 
            
            child1, child2 = parent1.copy(), parent2.copy()

            if n_features > 0 and np.random.rand() < crossover_rate:
                if n_features > 1: 
                    crossover_point = np.random.randint(1, n_features)
                    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            
            if n_features > 0:
                for j in range(n_features):
                    if np.random.rand() < mutation_rate:
                        child1[j] = not child1[j]
            offspring_population[i] = child1

            if i + 1 < n_population:
                if n_features > 0:
                    for j in range(n_features):
                        if np.random.rand() < mutation_rate:
                            child2[j] = not child2[j]
                offspring_population[i+1] = child2
        
        population = offspring_population

        end_time_gen = time.time()
        gen_time = end_time_gen - start_time_gen
        history_time_per_generation.append(gen_time)
        history_best_fitness_overall.append(best_fitness_overall if best_fitness_overall != -float('inf') else 0.0)
        history_gen_best_fitness.append(current_gen_best_fitness_val if current_gen_best_fitness_val != -float('inf') else 0.0)
        history_avg_fitness.append(np.mean(fitness_values_this_gen) if fitness_values_this_gen else 0.0)
        history_features_count.append(np.sum(best_solution_overall) if best_solution_overall is not None else 0)
        print(f"DEBUG GA: Fim da Geração {gen + 1}. MelhorFitnessGeral: {history_best_fitness_overall[-1]:.4f}, MelhorFitnessGen: {history_gen_best_fitness[-1]:.4f}, MédiaFitnessGen: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}, Tempo: {gen_time:.2f}s")

    print(f"DEBUG GA: Algoritmo Genético concluído. Melhor fitness final: {best_fitness_overall:.4f}")
    return {
        'best_solution_vector': best_solution_overall.tolist() if best_solution_overall is not None else ([] if n_features > 0 else []),
        'best_fitness': best_fitness_overall if best_fitness_overall != -float('inf') else -1.0,
        'selected_features_count': int(np.sum(best_solution_overall)) if best_solution_overall is not None else 0,
        'history': {
            'fitness_overall': history_best_fitness_overall,
            'fitness_epoch': history_gen_best_fitness,
            'avg_fitness_epoch': history_avg_fitness,
            'features_count': history_features_count,
            'time_per_epoch': history_time_per_generation
        }
    }

def particle_swarm_optimization(
    X_num_tensor: torch.Tensor, 
    X_img_tensor: torch.Tensor, 
    y_tensor: torch.Tensor,
    X_num_val: torch.Tensor,        # <- ADICIONAR
    X_img_val: torch.Tensor,        # <- ADICIONAR
    y_val: torch.Tensor,            # <- ADICIONAR
    fitness_fn: Callable, 
    n_features: int, 
    n_particles: int, 
    n_iterations: int,
    w: float, 
    c1: float, 
    c2: float, 
    use_image_data: bool, 
    device: str,
    fitness_fn_epochs: int
):
    print(f"\nDEBUG PSO: Iniciando Otimização por Enxame de Partículas. n_features={n_features}, n_partículas={n_particles}, n_iterações={n_iterations}, epochs_fitness={fitness_fn_epochs}, w={w}, c1={c1}, c2={c2}")
    particles_position = np.random.rand(n_particles, n_features) > 0.5 if n_features > 0 else np.empty((n_particles, 0), dtype=bool)
    particles_velocity = np.random.rand(n_particles, n_features) * 0.1 if n_features > 0 else np.empty((n_particles, 0), dtype=float)

    if n_features > 0:
        for i in range(n_particles):
            if not np.any(particles_position[i]):
                random_feature_index = np.random.randint(0, n_features)
                particles_position[i, random_feature_index] = True
                print(f"DEBUG PSO: Partícula {i} era toda zero, ativada feature {random_feature_index}")
    
    pbest_value = np.full(n_particles, -float('inf'))
    pbest_position = particles_position.copy() 
    
    gbest_position = None
    gbest_value = -float('inf')

    history_best_fitness_overall = []
    history_iter_best_fitness = []
    history_avg_fitness = []
    history_features_count = []
    history_time_per_iteration = []

    print("DEBUG PSO: Iniciando avaliação da Iteração 0 (inicial)")
    start_time_iter0 = time.time()
    initial_fitness_values = []
    current_iter_best_fitness_val = -float('inf')

    for i in range(n_particles):
        fitness = fitness_fn(
            individual_mask=particles_position[i] if n_features > 0 else None,
            X_num_train=X_num_tensor if n_features > 0 else None,     # <- CORRIGIDO
            X_img_train=X_img_tensor if use_image_data else None,     # <- CORRIGIDO
            y_train=y_tensor,                                         # <- CORRIGIDO
            X_num_val=X_num_val if n_features > 0 else None,         # <- NOVO
            X_img_val=X_img_val if use_image_data else None,         # <- NOVO
            y_val=y_val,                                             # <- NOVO
            device=device, 
            epochs=fitness_fn_epochs,
            algorithm_id="PSO"
        )
        initial_fitness_values.append(fitness)
        pbest_value[i] = fitness

        if fitness > current_iter_best_fitness_val:
            current_iter_best_fitness_val = fitness
        
        if fitness > gbest_value:
            gbest_value = fitness
            gbest_position = particles_position[i].copy()
            print(f"DEBUG PSO I0: Novo gbest_value: {gbest_value:.4f}, Posição: {gbest_position.astype(int) if n_features > 0 else 'N/A'}, Features: {np.sum(gbest_position) if gbest_position is not None else 0}")
        elif fitness == gbest_value and n_features > 0:
            if gbest_position is None or (np.sum(particles_position[i]) < np.sum(gbest_position)):
                gbest_position = particles_position[i].copy()
                print(f"DEBUG PSO I0: Mesmo gbest_value {gbest_value:.4f} mas com menos features. Posição: {gbest_position.astype(int)}, Features: {np.sum(gbest_position)}")
    
    if gbest_position is None and n_particles > 0 and n_features > 0 : 
        if len(initial_fitness_values) > 0:
             best_initial_idx = np.argmax(initial_fitness_values) if any(f != -float('inf') for f in initial_fitness_values) else 0
             gbest_position = particles_position[best_initial_idx].copy()
             gbest_value = initial_fitness_values[best_initial_idx]
             print(f"DEBUG PSO I0: gbest_position inicializado com partícula {best_initial_idx} devido a todos os fitness serem ruins. Fitness: {gbest_value:.4f}")


    history_best_fitness_overall.append(gbest_value if gbest_value != -float('inf') else 0.0)
    history_iter_best_fitness.append(current_iter_best_fitness_val if current_iter_best_fitness_val != -float('inf') else 0.0)
    history_avg_fitness.append(np.mean(initial_fitness_values) if initial_fitness_values else 0.0)
    history_features_count.append(np.sum(gbest_position) if gbest_position is not None else 0)
    iter0_time = time.time() - start_time_iter0
    history_time_per_iteration.append(iter0_time)
    print(f"DEBUG PSO: Fim da Iteração 0. GBestOverall: {history_best_fitness_overall[-1]:.4f}, MelhorFitnessIter: {history_iter_best_fitness[-1]:.4f}, MédiaFitnessIter: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}, Tempo: {iter0_time:.2f}s")

    for iteration in range(n_iterations):
        print(f"\nDEBUG PSO: Iniciando Iteração {iteration + 1}/{n_iterations}")
        start_time_iter = time.time()
        fitness_values_this_iter = []
        current_iter_best_fitness_val = -float('inf')

        for i in range(n_particles):
            if n_features > 0 and gbest_position is not None: 
                r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
                cognitive_velocity = c1 * r1 * (pbest_position[i].astype(float) - particles_position[i].astype(float))
                social_velocity = c2 * r2 * (gbest_position.astype(float) - particles_position[i].astype(float))
                particles_velocity[i] = w * particles_velocity[i] + cognitive_velocity + social_velocity
                
                sigmoid_val = 1 / (1 + np.exp(-particles_velocity[i])) 
                particles_position[i] = np.random.rand(n_features) < sigmoid_val
                
                if not np.any(particles_position[i]):
                    random_feature_index = np.random.randint(0, n_features)
                    particles_position[i, random_feature_index] = True
                    print(f"DEBUG PSO Iter {iteration+1}: Partícula {i} ficou toda zero após atualização, ativada feature {random_feature_index}")

            fitness = fitness_fn(
                individual_mask=particles_position[i] if n_features > 0 else None,
                X_num_train=X_num_tensor if n_features > 0 else None,     # <- CORRIGIDO
                X_img_train=X_img_tensor if use_image_data else None,     # <- CORRIGIDO
                y_train=y_tensor,                                         # <- CORRIGIDO
                X_num_val=X_num_val if n_features > 0 else None,         # <- NOVO
                X_img_val=X_img_val if use_image_data else None,         # <- NOVO
                y_val=y_val,                                             # <- NOVO
                device=device, 
                epochs=fitness_fn_epochs,
                algorithm_id="PSO"
            )
            fitness_values_this_iter.append(fitness)

            if fitness > current_iter_best_fitness_val:
                current_iter_best_fitness_val = fitness
            
            if fitness > pbest_value[i]:
                pbest_value[i] = fitness
                pbest_position[i] = particles_position[i].copy()
            
            if fitness > gbest_value:
                gbest_value = fitness
                gbest_position = particles_position[i].copy()
                print(f"DEBUG PSO Iter {iteration+1}: Novo gbest_value: {gbest_value:.4f}, Posição: {gbest_position.astype(int) if n_features > 0 else 'N/A'}, Features: {np.sum(gbest_position) if gbest_position is not None else 0}")
            elif fitness == gbest_value and n_features > 0:
                 if gbest_position is None or (np.sum(particles_position[i]) < np.sum(gbest_position)):
                    gbest_position = particles_position[i].copy()
                    print(f"DEBUG PSO Iter {iteration+1}: Mesmo gbest_value {gbest_value:.4f} mas com menos features. Posição: {gbest_position.astype(int)}, Features: {np.sum(gbest_position)}")
        
        end_time_iter = time.time()
        iter_time = end_time_iter - start_time_iter
        history_time_per_iteration.append(iter_time)
        history_best_fitness_overall.append(gbest_value if gbest_value != -float('inf') else 0.0)
        history_iter_best_fitness.append(current_iter_best_fitness_val if current_iter_best_fitness_val != -float('inf') else 0.0)
        history_avg_fitness.append(np.mean(fitness_values_this_iter) if fitness_values_this_iter else 0.0)
        history_features_count.append(np.sum(gbest_position) if gbest_position is not None else 0)
        print(f"DEBUG PSO: Fim da Iteração {iteration + 1}. GBestOverall: {history_best_fitness_overall[-1]:.4f}, MelhorFitnessIter: {history_iter_best_fitness[-1]:.4f}, MédiaFitnessIter: {history_avg_fitness[-1]:.4f}, Features: {history_features_count[-1]}, Tempo: {iter_time:.2f}s")

    print(f"DEBUG PSO: Otimização por Enxame de Partículas concluída. Melhor fitness final: {gbest_value:.4f}")
    return {
        'best_solution_vector': gbest_position.tolist() if gbest_position is not None else ([] if n_features > 0 else []),
        'best_fitness': gbest_value if gbest_value != -float('inf') else -1.0,
        'selected_features_count': int(np.sum(gbest_position)) if gbest_position is not None else 0,
        'history': {
            'fitness_overall': history_best_fitness_overall,
            'fitness_epoch': history_iter_best_fitness,
            'avg_fitness_epoch': history_avg_fitness,
            'features_count': history_features_count,
            'time_per_epoch': history_time_per_iteration
        }
    }