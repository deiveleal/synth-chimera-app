# -*- coding: utf-8 -*-
"""
Feature selection page module for Synth Chimera application.

This module contains the FeatureSelectionPage class that performs
feature selection using Genetic Algorithm (GA) and Particle Swarm Optimization (PSO).
"""

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from utils.cnn_fitness import evaluate_features
from utils.device_detection import get_available_device
from utils.optimization import genetic_algorithm, particle_swarm_optimization
from models.cnn_model import MultimodalCNN


class FeatureSelectionPage:
    """
    Feature Selection Page for the Streamlit app.
    
    This page allows users to perform feature selection using GA and PSO
    algorithms with baseline accuracy calculation for comparison.
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
        X_num_val=None,
        X_img_val=None,
        y_val=None,
        save_results_callback=None
    ):
        """
        Initialize the feature selection page.
        
        Args:
            num_pop: GA population size
            num_gen: GA number of generations
            num_part: PSO number of particles
            num_iter: PSO number of iterations
            X_num: Training numerical features
            X_img: Training image data
            y: Training labels
            X_num_val: Validation numerical features
            X_img_val: Validation image data
            y_val: Validation labels
            save_results_callback: Callback function to save results
        """
        self.save_results_callback = save_results_callback
        self.X_num_tensor = X_num
        self.X_img_tensor = X_img
        self.y_tensor = y
        self.X_num_val_tensor = X_num_val
        self.X_img_val_tensor = X_img_val
        self.y_val_tensor = y_val
        
        # Store algorithm parameters
        self.num_pop = num_pop
        self.num_gen = num_gen
        self.num_part = num_part
        self.num_iter = num_iter

        # Validate input data
        if not self._validate_input_data():
            return

        self.n_features_numerical = 0
        if self.X_num_tensor is not None:
            self.n_features_numerical = self.X_num_tensor.shape[1]

        # Detect device
        self.device = get_available_device()
        st.write(f"Device in FeatureSelectionPage: {self.device}")
        
        # Display dataset information
        self._display_dataset_info()

        self.use_image_for_evaluation = (self.X_img_tensor is not None)
        
        # Calculate baseline accuracy first
        self.baseline_accuracy = self._calculate_baseline_accuracy()
        
        # Run optimization algorithms
        self._run_optimization_algorithms()

    def _validate_input_data(self) -> bool:
        """Validate input data and display appropriate error messages."""
        if self.X_num_tensor is None and self.X_img_tensor is None:
            st.error("Input data (X_num or X_img) not provided for FeatureSelectionPage.")
            return False
        
        if self.y_tensor is None:
            st.error("Labels (y) not provided for FeatureSelectionPage.")
            return False
        
        if self.X_num_val_tensor is None and self.X_img_val_tensor is None:
            st.error("Validation data (X_num_val or X_img_val) not provided for FeatureSelectionPage.")
            return False
        
        if self.y_val_tensor is None:
            st.error("Validation labels (y_val) not provided for FeatureSelectionPage.")
            return False
        
        return True

    def _display_dataset_info(self) -> None:
        """Display information about training and validation datasets."""
        if self.X_num_tensor is not None:
            st.info(f"📊 Training data: {self.X_num_tensor.shape[0]} samples")
        if self.X_num_val_tensor is not None:
            st.info(f"📊 Validation data: {self.X_num_val_tensor.shape[0]} samples")

    def _calculate_baseline_accuracy(self) -> float:
        """
        Calculate baseline accuracy using all features.
        
        Returns:
            float: Baseline accuracy with all features
        """
        try:
            st.info("🔄 Calculating baseline accuracy with all features...")
            
            # Determine image shape
            if self.X_img_tensor is not None:
                image_shape = (self.X_img_tensor.shape[1], self.X_img_tensor.shape[2], self.X_img_tensor.shape[3])
            else:
                image_shape = None
            
            # Create model with all features
            baseline_model = MultimodalCNN(
                num_struct_features=self.X_num_tensor.shape[1] if self.X_num_tensor is not None else 0,
                image_input_shape=image_shape,
                num_classes=len(torch.unique(self.y_tensor)),
                use_image=(self.X_img_tensor is not None)
            ).to(self.device)
            
            # Train baseline model briefly
            optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            baseline_model.train()
            for epoch in range(15):  # Quick training for baseline
                optimizer.zero_grad()
                outputs = baseline_model(self.X_num_tensor, self.X_img_tensor)
                loss = criterion(outputs, self.y_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluate on validation set
            baseline_model.eval()
            with torch.no_grad():
                val_outputs = baseline_model(self.X_num_val_tensor, self.X_img_val_tensor)
                _, predicted = torch.max(val_outputs.data, 1)
                baseline_accuracy = (predicted == self.y_val_tensor).float().mean().item()
            
            st.success(f"📊 **Baseline Accuracy (All Features): {baseline_accuracy:.4f}**")
            return baseline_accuracy
            
        except Exception as e:
            st.error(f"Error calculating baseline accuracy: {e}")
            return 0.0

    def _run_optimization_algorithms(self) -> None:
        """Run both GA and PSO optimization algorithms."""
        # Run Genetic Algorithm
        self._run_genetic_algorithm()
        
        # Separator
        st.write("\n" + "─" * 50 + "\n")
        
        # Run Particle Swarm Optimization
        self._run_particle_swarm_optimization()

    def _run_genetic_algorithm(self) -> None:
        """Run Genetic Algorithm optimization."""
        st.subheader("Running Genetic Algorithm")
        ga_results = None
        
        if self.n_features_numerical > 0 or self.use_image_for_evaluation:
            with st.spinner(f"GA is optimizing features... (Generations: {self.num_gen}, Population: {self.num_pop})"):
                ga_raw_results_dict = genetic_algorithm(
                    X_num_tensor=self.X_num_tensor,
                    X_img_tensor=self.X_img_tensor,
                    y_tensor=self.y_tensor,
                    X_num_val=self.X_num_val_tensor,
                    X_img_val=self.X_img_val_tensor,
                    y_val=self.y_val_tensor,
                    fitness_fn=evaluate_features,
                    n_features=self.n_features_numerical,
                    n_population=int(self.num_pop),
                    n_generations=int(self.num_gen),
                    mutation_rate=0.3,
                    crossover_rate=0.8,
                    use_image_data=self.use_image_for_evaluation,
                    device=self.device,
                    fitness_fn_epochs=12
                )

            ga_results = ga_raw_results_dict
            if ga_results:
                # Add baseline accuracy to results
                ga_results['baseline_accuracy'] = self.baseline_accuracy
                ga_results['image_data_was_used_for_evaluation'] = self.use_image_for_evaluation

            self._display_algorithm_results("GA", ga_results)
        else:
            st.warning("GA not run: No numerical features to select and image data not available.")

        # Save GA results
        if self.save_results_callback:
            self.save_results_callback("GA", ga_results)

    def _run_particle_swarm_optimization(self) -> None:
        """Run Particle Swarm Optimization."""
        st.subheader("Running Particle Swarm Optimization")
        pso_results = None
        
        if self.n_features_numerical > 0 or self.use_image_for_evaluation:
            with st.spinner(f"PSO is optimizing features... (Iterations: {self.num_iter}, Particles: {self.num_part})"):
                pso_raw_results_dict = particle_swarm_optimization(
                    X_num_tensor=self.X_num_tensor,
                    X_img_tensor=self.X_img_tensor,
                    y_tensor=self.y_tensor,
                    X_num_val=self.X_num_val_tensor,
                    X_img_val=self.X_img_val_tensor,
                    y_val=self.y_val_tensor,
                    fitness_fn=evaluate_features,
                    n_features=self.n_features_numerical,
                    n_particles=int(self.num_part),
                    n_iterations=int(self.num_iter),
                    w=0.5,
                    c1=3.0,
                    c2=3.0,
                    use_image_data=self.use_image_for_evaluation,
                    device=self.device,
                    fitness_fn_epochs=12
                )

            pso_results = pso_raw_results_dict
            if pso_results:
                # Add baseline accuracy to results
                pso_results['baseline_accuracy'] = self.baseline_accuracy
                pso_results['image_data_was_used_for_evaluation'] = self.use_image_for_evaluation

            self._display_algorithm_results("PSO", pso_results)
        else:
            st.warning("PSO not run: No numerical features to select and image data not available.")

        # Save PSO results
        if self.save_results_callback:
            self.save_results_callback("PSO", pso_results)

    def _display_algorithm_results(self, algorithm_name: str, results: dict) -> None:
        """
        Display results for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm (GA or PSO)
            results: Results dictionary from the algorithm
        """
        if not results:
            st.warning(f"{algorithm_name} results not available.")
            return

        selected_numerical_mask = np.array(results['best_solution_vector'])
        selected_numerical_count = results['selected_features_count']
        best_fitness = results['best_fitness']
        
        # Calculate improvement over baseline
        improvement = best_fitness - self.baseline_accuracy
        improvement_percent = (improvement / self.baseline_accuracy) * 100 if self.baseline_accuracy > 0 else 0
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label=f"{algorithm_name} Best Fitness (Validation Accuracy)",
                value=f"{best_fitness:.4f}",
                delta=f"{improvement:+.4f} ({improvement_percent:+.1f}%)"
            )
        
        with col2:
            st.metric(
                label=f"Selected Numerical Features ({algorithm_name})",
                value=selected_numerical_count
            )

        # Additional information
        if self.use_image_for_evaluation:
            total_selected_for_display = selected_numerical_count + 1
            st.write(f"📸 Image data was considered in {algorithm_name} evaluation.")
            st.write(f"🔢 Total features effectively used by model ({algorithm_name}): {total_selected_for_display}")

        if self.n_features_numerical > 0:
            reduction_percentage = (1 - (selected_numerical_count / self.n_features_numerical)) * 100
            st.write(f"📉 Numerical Dimensionality Reduction ({algorithm_name}): {reduction_percentage:.2f}%")

        st.write(f"🎯 {algorithm_name}-selected numerical features (binary mask): {selected_numerical_mask.astype(int)}")
        
        # Show comparison with baseline
        if improvement > 0:
            st.success(f"✅ {algorithm_name} improved accuracy by {improvement:.4f} over baseline!")
        elif improvement < 0:
            st.warning(f"⚠️ {algorithm_name} accuracy decreased by {abs(improvement):.4f} compared to baseline.")
        else:
            st.info(f"ℹ️ {algorithm_name} achieved same accuracy as baseline.")
        
        st.info("🎯 Fitness calculated using validation data to avoid overfitting")


def main():
    """Main function for running the Feature Selection page standalone."""
    st.write("Feature Selection Page - requires dataset and parameters")


if __name__ == "__main__":
    main()