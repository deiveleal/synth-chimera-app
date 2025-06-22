import numpy as np
import streamlit as st

from utils.cnn_fitness import evaluate_features
from utils.device_detection import get_available_device
from utils.optimization import genetic_algorithm, particle_swarm_optimization


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
            X_num=None,
            X_img=None,
            y=None,
            save_results_callback=None):

        self.save_results_callback = save_results_callback
        self.X_num_tensor = X_num
        self.X_img_tensor = X_img
        self.y_tensor = y

        if self.X_num_tensor is None and self.X_img_tensor is None:
            st.error(
                "Dados de entrada (X_num ou X_img) não fornecidos para FeatureSelectionPage.")
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

        self.use_image_for_evaluation = (self.X_img_tensor is not None)

        # --- Algoritmo Genético ---
        st.subheader("Running Genetic Algorithm")
        ga_results = None
        if self.n_features_numerical > 0 or self.use_image_for_evaluation:
            with st.spinner(f"GA is optimizing features... (Generations: {num_gen}, Population: {num_pop})"):
                ga_raw_results_dict = genetic_algorithm(
                    X_num_tensor=self.X_num_tensor,
                    X_img_tensor=self.X_img_tensor,
                    y_tensor=self.y_tensor,
                    fitness_fn=evaluate_features,
                    n_features=self.n_features_numerical,
                    n_population=int(num_pop),
                    n_generations=int(num_gen),
                    mutation_rate=0.3,
                    crossover_rate=0.8,
                    use_image_data=self.use_image_for_evaluation,
                    device=self.device,
                    fitness_fn_epochs=12
                )

            ga_results = ga_raw_results_dict
            if ga_results:
                ga_results['image_data_was_used_for_evaluation'] = self.use_image_for_evaluation

            ga_selected_numerical_mask = np.array(
                ga_results['best_solution_vector'])
            ga_selected_numerical_count = ga_results['selected_features_count']
            ga_total_selected_for_display = ga_selected_numerical_count + \
                (1 if self.use_image_for_evaluation and ga_selected_numerical_count >= 0 else 0)

            st.write(
                f"GA Best Fitness (e.g., Loss): {ga_results['best_fitness']:.4f}")
            st.write(
                f"Number of selected numerical features (GA): {ga_selected_numerical_count}")
            if self.use_image_for_evaluation:
                st.write(f"Image data was considered in GA evaluation.")
            st.write(
                f"Total features effectively used by model (GA): {ga_total_selected_for_display}")

            if self.n_features_numerical > 0:
                reduction_percentage_ga = (
                    1 - (ga_selected_numerical_count / self.n_features_numerical)) * 100
                st.write(
                    f"Numerical Dimensionality Reduction (GA): {reduction_percentage_ga:.2f}%")

            st.write(
                f"GA-selected numerical features (binary mask): {ga_selected_numerical_mask.astype(int)}")
        else:
            st.warning(
                "GA not run: No numerical features to select and image data not available/used for evaluation context.")

        if self.save_results_callback and ga_results is not None:
            self.save_results_callback("GA", ga_results)
        elif self.save_results_callback:
            self.save_results_callback("GA", None)

        # --- Particle Swarm Optimization ---
        st.write("\n" + "--" * 30 + "\n")
        st.subheader("Running Particle Swarm Optimization")
        pso_results = None
        if self.n_features_numerical > 0 or self.use_image_for_evaluation:
            with st.spinner(f"PSO is optimizing features... (Iterations: {num_iter}, Particles: {num_part})"):
                pso_raw_results_dict = particle_swarm_optimization(
                    X_num_tensor=self.X_num_tensor,
                    X_img_tensor=self.X_img_tensor,
                    y_tensor=self.y_tensor,
                    fitness_fn=evaluate_features,
                    n_features=self.n_features_numerical,
                    n_particles=int(num_part),
                    n_iterations=int(num_iter),
                    w=0.5,
                    c1=3.0,
                    c2= 3.0,
                    use_image_data=self.use_image_for_evaluation,
                    device=self.device,
                    fitness_fn_epochs=12
                )

            pso_results = pso_raw_results_dict
            if pso_results:
                pso_results['image_data_was_used_for_evaluation'] = self.use_image_for_evaluation

            pso_selected_numerical_mask = np.array(
                pso_results['best_solution_vector'])
            pso_selected_numerical_count = pso_results['selected_features_count']
            pso_total_selected_for_display = pso_selected_numerical_count + \
                (1 if self.use_image_for_evaluation and pso_selected_numerical_count >= 0 else 0)

            st.write(
                f"PSO Best Fitness (e.g., Loss): {pso_results['best_fitness']:.4f}")
            st.write(
                f"Number of selected numerical features (PSO): {pso_selected_numerical_count}")
            if self.use_image_for_evaluation:
                st.write(f"Image data was considered in PSO evaluation.")
            st.write(
                f"Total features effectively used by model (PSO): {pso_total_selected_for_display}")

            if self.n_features_numerical > 0:
                reduction_percentage_pso = (
                    1 - (pso_selected_numerical_count / self.n_features_numerical)) * 100
                st.write(
                    f"Numerical Dimensionality Reduction (PSO): {reduction_percentage_pso:.2f}%")

            st.write(
                f"PSO-selected numerical features (binary mask): {pso_selected_numerical_mask.astype(int)}")
        else:
            st.warning(
                "PSO not run: No numerical features to select and image data not available/used for evaluation context.")

        if self.save_results_callback and pso_results is not None:
            self.save_results_callback("PSO", pso_results)
        elif self.save_results_callback:
            self.save_results_callback("PSO", None)


if __name__ == "__main__":
    FeatureSelectionPage()
