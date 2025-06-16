# import streamlit as st
# import torch
# import numpy as np

# from utils.cnn_fitness import evaluate_features
# from utils.device_detection import get_available_device
# from utils.generate_dataset import generate_multimodal_dataset
# from utils.hybrid_pso_ga_wrapper import hybrid_pso_ga_optimization
# from utils.optimization import genetic_algorithm, particle_swarm_optimization


# class FeatureSelectionPage:
#     """
#     Feature Selection Page for the Streamlit app.
#     This page allows users to perform feature selection using GA and PSO.
#     """

#     def __init__(
#             self,
#             num_pop=None,
#             num_gen=None,
#             num_part=None,
#             num_iter=None,
#             X_num=None,
#             X_img=None,
#             y=None,
#             save_results_callback=None): 

#         self.save_results_callback = save_results_callback
#         self.X_num = X_num
#         self.X_img = X_img
#         self.y = y
#         self.num_features = X_num.shape[1] + 1  # +1 para imagem
#         self.ga_selected_features = None
#         self.pso_selected_features = None

#         # Detect device
#         device = get_available_device()

#         use_image_flag = (self.X_img_tensor is not None)


#         # Fitness Function
#         def fitness_fn(X_selected, X_img, y, use_image): return evaluate_features(
#             X_selected, X_img, y, device, use_image)
        
#         # Criar nomes de features para visualização
#         feature_names = [f"Feature_{i+1}" for i in range(X_num.shape[1])]
#         feature_names.append("Image_Feature")


#         # GA Feature Selection
#         st.write("Executando Algoritmo Genético...")
#         self.ga_selected_features = genetic_algorithm(
#             X_num, X_img, y, fitness_fn, device=device, num_generations=num_gen, population_size=num_pop)
        
#         ga_fitness = fitness_fn(
#             X_num[:, self.ga_selected_features[:-1].astype(bool)],
#             X_img, y, self.ga_selected_features[-1].astype(bool))
        
#         ga_results = {
#             'best_solution': self.ga_selected_features.tolist(),
#             'best_fitness': float(ga_fitness),
#             'n_features': self.num_features,
#             # Se não tivermos históricos reais, criar simulações básicas
#             'best_fitness_history': [float(ga_fitness * 0.7), float(ga_fitness * 0.8), float(ga_fitness * 0.9), float(ga_fitness)],
#             'avg_fitness_history': [float(ga_fitness * 0.5), float(ga_fitness * 0.6), float(ga_fitness * 0.7), float(ga_fitness * 0.8)],
#             'feature_count_history': [int(sum(self.ga_selected_features)) for _ in range(4)],
#             'execution_times': [0.5, 0.5, 0.5, 0.5]
#         }
        
#         st.write(f"GA Fitness Score: {ga_fitness:.2f}")
#         st.write(
#             f"Number of selected features (GA): {self.ga_selected_features.sum()}")
#         st.write(
#             f"Dimentionality reduction (GA): {(1 - (self.ga_selected_features.sum() / self.ga_selected_features.shape[0])) * 100:.2f}%")
#         st.write(
#             f"GA-selected features (binary mask): {self.ga_selected_features.astype(int)}")
#         # Salvar resultados GA
#         if self.save_results_callback:
#             self.save_results_callback("GA", ga_results)



#         # PSO Feature Selection
#         st.write("\n")
#         st.write("--" * 50)
#         st.write("Executando PSO...")
#         self.pso_selected_features = particle_swarm_optimization(
#             X_num, X_img, y, fitness_fn, device=device, num_iterations=num_iter, num_particles=num_part)
        
#         pso_fitness = fitness_fn(
#             X_num[:, self.pso_selected_features[:-1].astype(bool)],
#             X_img, y, self.pso_selected_features[-1].astype(bool))
        
#         pso_results = {
#             'best_solution': self.pso_selected_features.tolist(),
#             'best_fitness': float(pso_fitness),
#             'n_features': self.num_features,
#             # Se não tivermos históricos reais, criar simulações básicas
#             'best_fitness_history': [float(pso_fitness * 0.7), float(pso_fitness * 0.8), float(pso_fitness * 0.9), float(pso_fitness)],
#             'avg_fitness_history': [float(pso_fitness * 0.5), float(pso_fitness * 0.6), float(pso_fitness * 0.7), float(pso_fitness * 0.8)],
#             'feature_count_history': [int(sum(self.pso_selected_features)) for _ in range(4)],
#             'execution_times': [0.5, 0.5, 0.5, 0.5]
#         }
#         # Salvar resultados PSO
#         if self.save_results_callback:
#             self.save_results_callback("PSO", pso_results)

#         # Exibir resultados PSO        
#         st.write(f"PSO Fitness Score: {pso_fitness:.2f}")
#         st.write(
#             f"Number of selected features (PSO): {self.pso_selected_features.sum()}")
#         st.write(
#             f"Dimentionality reduction (PSO): {(1 - (self.pso_selected_features.sum() / self.pso_selected_features.shape[0])) * 100:.2f}%")
#         st.write(
#             f"PSO-selected features (binary mask): {self.pso_selected_features.astype(int)}")
#         # Salvar resultados PSO
#         if self.save_results_callback:
#             self.save_results_callback("PSO", pso_results)


        # # Hybrid PSO-GA Feature Selection
        # st.write("\n")
        # st.write("--" * 50)
        # st.write("Executando Algoritmo Híbrido PSO-GA...")

        # self.hybrid_results = hybrid_pso_ga_optimization(
        #     X_num, X_img, y, fitness_fn, device=device, num_generations=num_gen, population_size=num_pop)
        
        # # Extrair a máscara binária e garantir que seja numpy array, não tensor
        # if isinstance(self.hybrid_results, dict):
        #     self.hybrid_selected_features = np.array(self.hybrid_results['best_solution'])
        # else:
        #     self.hybrid_selected_features = self.hybrid_results.cpu().numpy() if isinstance(self.hybrid_results, torch.Tensor) else np.array(self.hybrid_results)
        
        # hybrid_fitness = fitness_fn(
        #     X_num[:, self.hybrid_selected_features[:-1].astype(bool)],
        #     X_img, y, bool(self.hybrid_selected_features[-1]))

        # # Exibir resultados Híbrido PSO-GA
        # st.write(f"Hybrid PSO-GA Fitness Score: {hybrid_fitness:.2f}")
        # st.write(
        #     f"Number of selected features (Hybrid PSO-GA): {np.sum(self.hybrid_selected_features)}")
        # st.write(
        #     f"Dimentionality reduction (Hybrid PSO-GA): {(1 - (np.sum(self.hybrid_selected_features) / self.hybrid_selected_features.shape[0])) * 100:.2f}%")
        # st.write(
        #     f"Hybrid PSO-GA-selected features (binary mask): {self.hybrid_selected_features.astype(int)}")
        
        # Se o fitness calculado for diferente do retornado pelo algoritmo, atualizar
        # if isinstance(self.hybrid_results, dict) and 'best_fitness' in self.hybrid_results:
        #     self.hybrid_results['best_fitness'] = float(hybrid_fitness)
        
        # # Salvar resultados Híbridos
        # if self.save_results_callback:
        #     self.save_results_callback("HYBRID", self.hybrid_results)

import streamlit as st
import torch
import numpy as np

from utils.cnn_fitness import evaluate_features # Função de fitness real
from utils.device_detection import get_available_device
# from utils.generate_dataset import generate_multimodal_dataset # Não usado diretamente aqui
# from utils.hybrid_pso_ga_wrapper import hybrid_pso_ga_optimization # Comentado no original
from utils.optimization import genetic_algorithm, particle_swarm_optimization # Algoritmos reais

class FeatureSelectionPage:
    """
    Feature Selection Page for the Streamlit app.
    This page allows users to perform feature selection using GA and PSO.
    """

    def __init__(
            self,
            num_pop=None,
            num_gen=None,
            num_part=None,
            num_iter=None,
            X_num=None, # Tensor de features numéricas
            X_img=None, # Tensor de features de imagem
            y=None,     # Tensor de labels
            save_results_callback=None):

        self.save_results_callback = save_results_callback
        self.X_num_tensor = X_num # Renomeado para clareza, já deve ser tensor
        self.X_img_tensor = X_img # Renomeado para clareza, já deve ser tensor
        self.y_tensor = y         # Renomeado para clareza, já deve ser tensor

        if self.X_num_tensor is None and self.X_img_tensor is None:
            st.error("Dados de entrada (X_num ou X_img) não fornecidos para FeatureSelectionPage.")
            return
        if self.y_tensor is None:
            st.error("Labels (y) não fornecidos para FeatureSelectionPage.")
            return

        self.n_features_numerical = 0
        if self.X_num_tensor is not None:
            self.n_features_numerical = self.X_num_tensor.shape[1]
        
        # Detect device
        self.device = get_available_device()
        st.write(f"Device in FeatureSelectionPage: {self.device}")


        # Determinar se os dados de imagem devem ser usados na avaliação de fitness
        # Esta flag será passada para os algoritmos de otimização.
        self.use_image_for_evaluation = (self.X_img_tensor is not None)

        # --- Algoritmo Genético ---
        st.subheader("Running Genetic Algorithm")
        ga_results = None # Inicializar
        if self.n_features_numerical > 0 or self.use_image_for_evaluation: # Só executa se houver o que selecionar/avaliar
            with st.spinner(f"GA is optimizing features... (Generations: {num_gen}, Population: {num_pop})"):
                # Chamar o algoritmo genético real, que retorna um dicionário com histórico
                ga_raw_results_dict = genetic_algorithm(
                    X_num_tensor=self.X_num_tensor, # Pode ser None se n_features_numerical == 0
                    X_img_tensor=self.X_img_tensor, # Passado independentemente, evaluate_features decide
                    y_tensor=self.y_tensor,
                    fitness_fn=evaluate_features, # Passa a função de fitness real
                    n_features=self.n_features_numerical, # GA otimiza a máscara para features numéricas
                    n_population=int(num_pop),
                    n_generations=int(num_gen),
                    mutation_rate=0.1,  # Exemplo, pode ser configurável
                    crossover_rate=0.8, # Exemplo
                    use_image_data=self.use_image_for_evaluation, # Informa evaluate_features se a imagem está disponível/deve ser usada
                    device=self.device
                )
            
            # Usar os resultados reais, incluindo o histórico
            ga_results = ga_raw_results_dict 
            if ga_results: 
                ga_results['image_data_was_used_for_evaluation'] = self.use_image_for_evaluation
 
            
            # Exibir informações baseadas nos resultados reais
            ga_selected_numerical_mask = np.array(ga_results['best_solution_vector'])
            ga_selected_numerical_count = ga_results['selected_features_count']
            ga_total_selected_for_display = ga_selected_numerical_count + (1 if self.use_image_for_evaluation and ga_selected_numerical_count >= 0 else 0) # Adiciona 1 se imagem foi usada e houve seleção válida

            st.write(f"GA Best Fitness (e.g., Loss): {ga_results['best_fitness']:.4f}")
            st.write(f"Number of selected numerical features (GA): {ga_selected_numerical_count}")
            if self.use_image_for_evaluation:
                 st.write(f"Image data was considered in GA evaluation.")
            st.write(f"Total features effectively used by model (GA): {ga_total_selected_for_display}")
            
            if self.n_features_numerical > 0:
                reduction_percentage_ga = (1 - (ga_selected_numerical_count / self.n_features_numerical)) * 100
                st.write(f"Numerical Dimensionality Reduction (GA): {reduction_percentage_ga:.2f}%")
            
            st.write(f"GA-selected numerical features (binary mask): {ga_selected_numerical_mask.astype(int)}")
        else:
            st.warning("GA not run: No numerical features to select and image data not available/used for evaluation context.")
        
        if self.save_results_callback and ga_results is not None:
            self.save_results_callback("GA", ga_results)
        elif self.save_results_callback: # Salvar None se não rodou
            self.save_results_callback("GA", None)


        # --- Particle Swarm Optimization ---
        st.write("\n" + "--" * 30 + "\n")
        st.subheader("Running Particle Swarm Optimization")
        pso_results = None # Inicializar
        if self.n_features_numerical > 0 or self.use_image_for_evaluation:
            with st.spinner(f"PSO is optimizing features... (Iterations: {num_iter}, Particles: {num_part})"):
                # Chamar o PSO real, que retorna um dicionário com histórico
                pso_raw_results_dict = particle_swarm_optimization(
                    X_num_tensor=self.X_num_tensor,
                    X_img_tensor=self.X_img_tensor,
                    y_tensor=self.y_tensor,
                    fitness_fn=evaluate_features, # Passa a função de fitness real
                    n_features=self.n_features_numerical, # PSO otimiza a máscara para features numéricas
                    n_particles=int(num_part),
                    n_iterations=int(num_iter),
                    w=0.5,  # Exemplo
                    c1=1.5, # Exemplo
                    c2=1.5, # Exemplo
                    use_image_data=self.use_image_for_evaluation,
                    device=self.device
                )

            # Usar os resultados reais, incluindo o histórico
            pso_results = pso_raw_results_dict
            if pso_results: 
                pso_results['image_data_was_used_for_evaluation'] = self.use_image_for_evaluation


            pso_selected_numerical_mask = np.array(pso_results['best_solution_vector'])
            pso_selected_numerical_count = pso_results['selected_features_count']
            pso_total_selected_for_display = pso_selected_numerical_count + (1 if self.use_image_for_evaluation and pso_selected_numerical_count >=0 else 0)

            st.write(f"PSO Best Fitness (e.g., Loss): {pso_results['best_fitness']:.4f}")
            st.write(f"Number of selected numerical features (PSO): {pso_selected_numerical_count}")
            if self.use_image_for_evaluation:
                 st.write(f"Image data was considered in PSO evaluation.")
            st.write(f"Total features effectively used by model (PSO): {pso_total_selected_for_display}")

            if self.n_features_numerical > 0:
                reduction_percentage_pso = (1 - (pso_selected_numerical_count / self.n_features_numerical)) * 100
                st.write(f"Numerical Dimensionality Reduction (PSO): {reduction_percentage_pso:.2f}%")

            st.write(f"PSO-selected numerical features (binary mask): {pso_selected_numerical_mask.astype(int)}")
        else:
            st.warning("PSO not run: No numerical features to select and image data not available/used for evaluation context.")

        if self.save_results_callback and pso_results is not None:
            self.save_results_callback("PSO", pso_results)
        elif self.save_results_callback: # Salvar None se não rodou
            self.save_results_callback("PSO", None)

if __name__ == "__main__":
    FeatureSelectionPage()
