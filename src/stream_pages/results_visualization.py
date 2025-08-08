"""
Results visualization page module for Synth Chimera application.

This module contains the ResultsVisualizationPage class that displays
comparative results and visualizations for GA and PSO feature selection algorithms.
"""

import streamlit as st
import plotly.graph_objects as go  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Optional, List, Dict, Any


class ResultsVisualizationPage:
    """
    Results visualization page for comparing GA and PSO optimization results.
    
    This class creates comprehensive visualizations and metrics displays for
    analyzing the performance of Genetic Algorithm (GA) and Particle Swarm
    Optimization (PSO) feature selection results.
    
    Args:
        ga_results: Results dictionary from Genetic Algorithm execution
        pso_results: Results dictionary from PSO execution
        numerical_feature_names: List of original numerical feature names
    """
    
    def __init__(
        self, 
        ga_results: Optional[Dict[str, Any]], 
        pso_results: Optional[Dict[str, Any]], 
        numerical_feature_names: Optional[List[str]] = None
    ):
        """Initialize the results visualization page."""
        self.ga_results = ga_results
        self.pso_results = pso_results
        self.numerical_feature_names = numerical_feature_names if numerical_feature_names else []
        
        # Determine total number of original numerical features
        self.total_numerical_features_original = len(self.numerical_feature_names)
        if self.total_numerical_features_original == 0:
            if self.ga_results and 'best_solution_vector' in self.ga_results:
                self.total_numerical_features_original = len(self.ga_results['best_solution_vector'])
            elif self.pso_results and 'best_solution_vector' in self.pso_results:
                self.total_numerical_features_original = len(self.pso_results['best_solution_vector'])

        self._render_page()
    
    def _render_page(self) -> None:
        """Render the complete results visualization page."""
        st.header("Comparative Results Visualization")
        
        if not self.ga_results and not self.pso_results:
            st.warning("No results available for visualization. Run feature selection first.")
            return

        # Create tabs for different visualization sections
        tabs_list = [
            "Convergence", 
            "Selected Features Count", 
            "Performance Metrics", 
            "Feature Selection Details"
        ]
        
        tabs = st.tabs(tabs_list)

        with tabs[0]:
            self.plot_convergence()
        with tabs[1]:
            self.plot_selected_features_count_over_time()
        with tabs[2]:
            self.display_performance_metrics()
        with tabs[3]:
            self.display_feature_selection_details()
    
    def _get_history_data(self, results: Optional[Dict[str, Any]], key: str) -> List[Any]:
        """
        Extract history data from results dictionary.
        
        Args:
            results: Results dictionary containing history data
            key: Key to extract from history
            
        Returns:
            List of historical values or empty list if not found
        """
        if (results and 'history' in results and 
            isinstance(results['history'], dict) and key in results['history']):
            return results['history'][key]
        return []

    def plot_convergence(self) -> None:
        """Plot fitness convergence charts for GA and PSO algorithms."""
        st.subheader("Fitness Convergence Across Generations/Iterations")
        fig = go.Figure()
        
        # GA Data
        ga_fitness_overall_history = self._get_history_data(self.ga_results, 'fitness_overall')
        ga_fitness_epoch_history = self._get_history_data(self.ga_results, 'fitness_epoch')
        ga_avg_fitness_epoch_history = self._get_history_data(self.ga_results, 'avg_fitness_epoch')
        
        if ga_fitness_overall_history:
            epochs_ga = list(range(len(ga_fitness_overall_history)))
            fig.add_trace(go.Scatter(
                x=epochs_ga, 
                y=ga_fitness_overall_history,
                mode='lines+markers', 
                name='Best Global Fitness (GA)',
                line=dict(color='blue', width=2)
            ))
            
            if ga_fitness_epoch_history and len(ga_fitness_epoch_history) == len(epochs_ga):
                fig.add_trace(go.Scatter(
                    x=epochs_ga, 
                    y=ga_fitness_epoch_history, 
                    mode='lines', 
                    name='Best Generation Fitness (GA)', 
                    line=dict(color='cyan', width=1, dash='dot')
                ))
                
            if ga_avg_fitness_epoch_history and len(ga_avg_fitness_epoch_history) == len(epochs_ga):
                fig.add_trace(go.Scatter(
                    x=epochs_ga, 
                    y=ga_avg_fitness_epoch_history, 
                    mode='lines', 
                    name='Average Generation Fitness (GA)', 
                    line=dict(color='lightskyblue', width=1, dash='dash')
                ))

        # PSO Data
        pso_fitness_overall_history = self._get_history_data(self.pso_results, 'fitness_overall')
        pso_fitness_epoch_history = self._get_history_data(self.pso_results, 'fitness_epoch')
        pso_avg_fitness_epoch_history = self._get_history_data(self.pso_results, 'avg_fitness_epoch')

        if pso_fitness_overall_history:
            epochs_pso = list(range(len(pso_fitness_overall_history)))
            fig.add_trace(go.Scatter(
                x=epochs_pso, 
                y=pso_fitness_overall_history,
                mode='lines+markers', 
                name='Best Global Fitness (PSO)',
                line=dict(color='red', width=2)
            ))
            
            if pso_fitness_epoch_history and len(pso_fitness_epoch_history) == len(epochs_pso):
                fig.add_trace(go.Scatter(
                    x=epochs_pso, 
                    y=pso_fitness_epoch_history, 
                    mode='lines', 
                    name='Best Iteration Fitness (PSO)', 
                    line=dict(color='magenta', width=1, dash='dot')
                ))
                
            if pso_avg_fitness_epoch_history and len(pso_avg_fitness_epoch_history) == len(epochs_pso):
                fig.add_trace(go.Scatter(
                    x=epochs_pso, 
                    y=pso_avg_fitness_epoch_history, 
                    mode='lines', 
                    name='Average Iteration Fitness (PSO)', 
                    line=dict(color='lightcoral', width=1, dash='dash')
                ))
        
        if not fig.data:
            st.write(
                "No convergence data to plot. Please verify that optimization "
                "algorithms recorded fitness history correctly."
            )
        else:
            fig.update_layout(
                xaxis_title="Generation / Iteration",
                yaxis_title="Fitness Value",
                legend_title_text='Fitness Metrics',
                yaxis_range=[0, 1.05]
            )
            st.plotly_chart(fig, use_container_width=True)

    def plot_selected_features_count_over_time(self) -> None:
        """Plot the number of selected numerical features over time."""
        st.subheader("Number of Selected Numerical Features Over Generations/Iterations")
        fig = go.Figure()

        ga_features_history = self._get_history_data(self.ga_results, 'features_count')
        if ga_features_history:
            generations_ga = list(range(len(ga_features_history)))
            fig.add_trace(go.Scatter(
                x=generations_ga, 
                y=ga_features_history,
                mode='lines+markers', 
                name='Selected Features (GA)'
            ))

        pso_features_history = self._get_history_data(self.pso_results, 'features_count')
        if pso_features_history:
            iterations_pso = list(range(len(pso_features_history)))
            fig.add_trace(go.Scatter(
                x=iterations_pso, 
                y=pso_features_history,
                mode='lines+markers', 
                name='Selected Features (PSO)'
            ))
        
        # Add reference line for total original features
        if self.total_numerical_features_original > 0:
            max_gens_iters = 0
            if ga_features_history: 
                max_gens_iters = max(max_gens_iters, len(ga_features_history))
            if pso_features_history: 
                max_gens_iters = max(max_gens_iters, len(pso_features_history))
            
            if max_gens_iters > 0:
                fig.add_trace(go.Scatter(
                    x=list(range(max_gens_iters)), 
                    y=[self.total_numerical_features_original] * max_gens_iters,
                    mode='lines', 
                    name='Total Original Numerical Features', 
                    line=dict(dash='dot', color='grey')
                ))

        if not fig.data:
            st.write("No selected features count data to plot.")
        else:
            fig.update_layout(
                xaxis_title="Generation / Iteration", 
                yaxis_title="Number of Selected Numerical Features"
            )
            st.plotly_chart(fig, use_container_width=True)

    def display_performance_metrics(self) -> None:
        """Display final performance metrics for both algorithms."""
        st.subheader("Final Performance Metrics")
        data = []
        reduction_ga_text = "N/A"
        reduction_pso_text = "N/A"
        
        # Get baseline accuracy (accuracy with all features)
        baseline_accuracy = None
        if self.ga_results and 'baseline_accuracy' in self.ga_results:
            baseline_accuracy = self.ga_results['baseline_accuracy']
        elif self.pso_results and 'baseline_accuracy' in self.pso_results:
            baseline_accuracy = self.pso_results['baseline_accuracy']

        if self.ga_results:
            num_selected_ga = self.ga_results.get('selected_features_count', 0)
            if self.total_numerical_features_original > 0 and isinstance(num_selected_ga, int):
                reduction_ga = (
                    (self.total_numerical_features_original - num_selected_ga) / 
                    self.total_numerical_features_original
                ) * 100
                reduction_ga_text = (
                    f"{reduction_ga:.1f}% "
                    f"({self.total_numerical_features_original - num_selected_ga} features reduced)"
                )
            elif self.total_numerical_features_original == 0:
                reduction_ga_text = "N/A (no numerical features)"
            else:
                reduction_ga_text = "Error calculating GA reduction"
            
            ga_hist_len = len(self._get_history_data(self.ga_results, 'fitness_overall'))
            data.append({
                "Algorithm": "GA",
                "Final Accuracy (Selected Features)": f"{self.ga_results.get('best_fitness', -1.0):.4f}",
                "Baseline Accuracy (All Features)": f"{baseline_accuracy:.4f}" if baseline_accuracy is not None else "N/A",
                "Selected Numerical Features": num_selected_ga,
                "Total Generations Executed": ga_hist_len - 1 if ga_hist_len > 0 else 0
            })
        
        if self.pso_results:
            num_selected_pso = self.pso_results.get('selected_features_count', 0)
            if self.total_numerical_features_original > 0 and isinstance(num_selected_pso, int):
                reduction_pso = (
                    (self.total_numerical_features_original - num_selected_pso) / 
                    self.total_numerical_features_original
                ) * 100
                reduction_pso_text = (
                    f"{reduction_pso:.1f}% "
                    f"({self.total_numerical_features_original - num_selected_pso} features reduced)"
                )
            elif self.total_numerical_features_original == 0:
                reduction_pso_text = "N/A (no numerical features)"
            else:
                reduction_pso_text = "Error calculating PSO reduction"

            pso_hist_len = len(self._get_history_data(self.pso_results, 'fitness_overall'))
            data.append({
                "Algorithm": "PSO",
                "Final Accuracy (Selected Features)": f"{self.pso_results.get('best_fitness', -1.0):.4f}",
                "Baseline Accuracy (All Features)": f"{baseline_accuracy:.4f}" if baseline_accuracy is not None else "N/A",
                "Selected Numerical Features": num_selected_pso,
                "Total Iterations Executed": pso_hist_len - 1 if pso_hist_len > 0 else 0
            })
        
        if data:
            df_metrics = pd.DataFrame(data)
            st.table(df_metrics.set_index("Algorithm"))

            # Display baseline accuracy prominently if available
            if baseline_accuracy is not None:
                st.markdown("---")
                st.info(f"📊 **Baseline Accuracy (All Features):** {baseline_accuracy:.4f}")

            st.markdown("---")
            st.subheader("Detailed Final Accuracy")
            
            num_cols = 0
            if self.ga_results: num_cols += 1
            if self.pso_results: num_cols += 1
            
            if num_cols > 0:
                cols_accuracy = st.columns(num_cols)
                current_col = 0
                if self.ga_results:
                    with cols_accuracy[current_col]:
                        ga_accuracy = self.ga_results.get('best_fitness', -1.0)
                        delta_ga = None
                        if baseline_accuracy is not None:
                            delta_ga = ga_accuracy - baseline_accuracy
                        
                        st.metric(
                            label="GA Final Accuracy (Selected Features)", 
                            value=f"{ga_accuracy:.4f}",
                            delta=f"{delta_ga:.4f}" if delta_ga is not None else None,
                            delta_color="normal"
                        )
                    current_col += 1
                
                if self.pso_results:
                    with cols_accuracy[current_col]:
                        pso_accuracy = self.pso_results.get('best_fitness', -1.0)
                        delta_pso = None
                        if baseline_accuracy is not None:
                            delta_pso = pso_accuracy - baseline_accuracy
                        
                        st.metric(
                            label="PSO Final Accuracy (Selected Features)", 
                            value=f"{pso_accuracy:.4f}",
                            delta=f"{delta_pso:.4f}" if delta_pso is not None else None,
                            delta_color="normal"
                        )
            
            st.markdown("---") 
            st.subheader("Complexity Reduction (Numerical Features)")
            if num_cols > 0:
                cols_reduction = st.columns(num_cols)
                current_col = 0
                if self.ga_results:
                    with cols_reduction[current_col]:
                        st.metric(label="GA Reduction", value=reduction_ga_text)
                    current_col += 1
                if self.pso_results:
                    with cols_reduction[current_col]:
                        st.metric(label="PSO Reduction", value=reduction_pso_text)
            else:
                st.write("No reduction metrics to display.")
        else:
            st.write("No performance metrics to display.")
        
    def display_feature_selection_details(self) -> None:
        """Display detailed information about selected features."""
        st.subheader("Selected Numerical Features Details (Final Solution)")
        
        # GA Results
        if self.ga_results and 'best_solution_vector' in self.ga_results:
            st.markdown("#### Genetic Algorithm (GA)")
            ga_mask = np.array(self.ga_results['best_solution_vector'])
            st.markdown(f"**Binary Mask:** `{ga_mask.astype(int)}`")
            
            ga_selected_indices = self.ga_results.get('selected_features_indices')
            if ga_selected_indices is not None:
                st.markdown(f"**Selected Indices:** `{ga_selected_indices}`")
                if (self.numerical_feature_names and 
                    self.total_numerical_features_original == len(ga_mask)):
                    selected_names = [
                        self.numerical_feature_names[i] for i in ga_selected_indices 
                        if i < len(self.numerical_feature_names)
                    ]
                    if selected_names:
                        st.markdown("**Selected Feature Names:**")
                        st.text(", ".join(selected_names))
                    else:
                        st.markdown(
                            "*No numerical features selected with corresponding names "
                            "or empty names list.*"
                        )
                elif not self.numerical_feature_names:
                    st.markdown(
                        "*Original numerical feature names not provided for display.*"
                    )
            else:
                st.markdown("*Selected feature indices not available in GA results.*")
            st.markdown("---")

        # PSO Results
        if self.pso_results and 'best_solution_vector' in self.pso_results:
            st.markdown("#### Particle Swarm Optimization (PSO)")
            pso_mask = np.array(self.pso_results['best_solution_vector'])
            st.markdown(f"**Binary Mask:** `{pso_mask.astype(int)}`")

            pso_selected_indices = self.pso_results.get('selected_features_indices')
            if pso_selected_indices is not None:
                st.markdown(f"**Selected Indices:** `{pso_selected_indices}`")
                if (self.numerical_feature_names and 
                    self.total_numerical_features_original == len(pso_mask)):
                    selected_names = [
                        self.numerical_feature_names[i] for i in pso_selected_indices 
                        if i < len(self.numerical_feature_names)
                    ]
                    if selected_names:
                        st.markdown("**Selected Feature Names:**")
                        st.text(", ".join(selected_names))
                    else:
                        st.markdown(
                            "*No numerical features selected with corresponding names "
                            "or empty names list.*"
                        )
                elif not self.numerical_feature_names:
                    st.markdown(
                        "*Original numerical feature names not provided for display.*"
                    )
            else:
                st.markdown("*Selected feature indices not available in PSO results.*")
            st.markdown("---")
        
        if (not (self.ga_results and 'best_solution_vector' in self.ga_results) and 
            not (self.pso_results and 'best_solution_vector' in self.pso_results)):
            st.write("No feature selection details to display.")

        # Summary table with all collected data
        self._display_experiment_summary_table()
    
    def _display_experiment_summary_table(self) -> None:
        """Display comprehensive experiment summary table."""
        st.markdown("---")
        st.subheader("Experiment Summary Table")
        
        # Add validation information in header
        st.info("📊 All fitness metrics were calculated using validation data")

        # Create DataFrame with all historical data
        all_data = []

        if self.ga_results and 'history' in self.ga_results:
            ga_history = self.ga_results['history']
            max_len_ga = max([len(ga_history.get(key, [])) for key in ga_history.keys()] + [0])
            
            for i in range(max_len_ga):
                row = {
                    'Algorithm': 'GA',
                    'Generation/Iteration': i,
                    'Global Fitness': (
                        ga_history.get('fitness_overall', [])[i] 
                        if i < len(ga_history.get('fitness_overall', [])) else None
                    ),
                    'Epoch Fitness': (
                        ga_history.get('fitness_epoch', [])[i] 
                        if i < len(ga_history.get('fitness_epoch', [])) else None
                    ),
                    'Average Epoch Fitness': (
                        ga_history.get('avg_fitness_epoch', [])[i] 
                        if i < len(ga_history.get('avg_fitness_epoch', [])) else None
                    ),
                    'Selected Features': (
                        ga_history.get('features_count', [])[i] 
                        if i < len(ga_history.get('features_count', [])) else None
                    )
                }
                all_data.append(row)

        if self.pso_results and 'history' in self.pso_results:
            pso_history = self.pso_results['history']
            max_len_pso = max([len(pso_history.get(key, [])) for key in pso_history.keys()] + [0])
            
            for i in range(max_len_pso):
                row = {
                    'Algorithm': 'PSO',
                    'Generation/Iteration': i,
                    'Global Fitness': (
                        pso_history.get('fitness_overall', [])[i] 
                        if i < len(pso_history.get('fitness_overall', [])) else None
                    ),
                    'Epoch Fitness': (
                        pso_history.get('fitness_epoch', [])[i] 
                        if i < len(pso_history.get('fitness_epoch', [])) else None
                    ),
                    'Average Epoch Fitness': (
                        pso_history.get('avg_fitness_epoch', [])[i] 
                        if i < len(pso_history.get('avg_fitness_epoch', [])) else None
                    ),
                    'Selected Features': (
                        pso_history.get('features_count', [])[i] 
                        if i < len(pso_history.get('features_count', [])) else None
                    )
                }
                all_data.append(row)

        if all_data:
            df_all_data = pd.DataFrame(all_data)
            
            # Filter data by algorithm
            algorithm_filter = st.selectbox(
                "Filter by Algorithm:", 
                ["All", "GA", "PSO"]
            )
            
            if algorithm_filter != "All":
                df_filtered = df_all_data[df_all_data['Algorithm'] == algorithm_filter]
            else:
                df_filtered = df_all_data
            
            # Show basic statistics
            if not df_filtered.empty:
                st.markdown("**Data Statistics:**")
                cols_stats = st.columns(3)
                with cols_stats[0]:
                    st.metric("Total Records", len(df_filtered))
                with cols_stats[1]:
                    if 'Global Fitness' in df_filtered.columns:
                        max_fitness = df_filtered['Global Fitness'].max()
                        st.metric(
                            "Best Global Fitness", 
                            f"{max_fitness:.4f}" if pd.notna(max_fitness) else "N/A"
                        )
                with cols_stats[2]:
                    if 'Selected Features' in df_filtered.columns:
                        avg_features = df_filtered['Selected Features'].mean()
                        st.metric(
                            "Average Features", 
                            f"{avg_features:.1f}" if pd.notna(avg_features) else "N/A"
                        )
                
                # Display table
                st.dataframe(df_filtered, use_container_width=True, height=400)
                
                # Download option
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    label="📥 Download data as CSV",
                    data=csv,
                    file_name=f"execution_data_{algorithm_filter.lower()}.csv",
                    mime="text/csv"
                )
            else:
                st.write("No data available for the selected filter.")
        else:
            st.write("No historical data available to display.")


def main():
    """Main function for running the Results Visualization page standalone."""
    # This would normally be called with actual results data
    st.write("Results Visualization Page - requires GA and PSO results data")


if __name__ == "__main__":
    main()