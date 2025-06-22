import streamlit as st
import plotly.graph_objects as go # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore


class ResultsVisualizationPage:
    def __init__(self, ga_results, pso_results, numerical_feature_names=None):
        self.ga_results = ga_results
        self.pso_results = pso_results
        self.numerical_feature_names = numerical_feature_names if numerical_feature_names else []
        
        self.total_numerical_features_original = len(self.numerical_feature_names)
        if self.total_numerical_features_original == 0:
            if self.ga_results and 'best_solution_vector' in self.ga_results:
                self.total_numerical_features_original = len(self.ga_results['best_solution_vector'])
            elif self.pso_results and 'best_solution_vector' in self.pso_results:
                self.total_numerical_features_original = len(self.pso_results['best_solution_vector'])

        st.header("Visualização Comparativa de Resultados")
        
        if not self.ga_results and not self.pso_results:
            st.warning("Nenhum resultado disponível para visualização. Execute a seleção de features primeiro.")
            return

        tabs_list = ["Convergência", "Contagem de Features Selecionadas", "Métricas de Desempenho", "Detalhes da Seleção de Features"]
        
        tabs = st.tabs(tabs_list)

        with tabs[0]:
            self.plot_convergence()
        with tabs[1]:
            self.plot_selected_features_count_over_time()
        with tabs[2]:
            self.display_performance_metrics()
        with tabs[3]:
            self.display_feature_selection_details()
            
    def _get_history_data(self, results, key):
        if results and 'history' in results and isinstance(results['history'], dict) and key in results['history']:
            return results['history'][key]
        return []

    def plot_convergence(self):
        st.subheader("Convergência do Fitness ao Longo das Gerações/Iterações")
        fig = go.Figure()
        
        # GA Data
        ga_fitness_overall_history = self._get_history_data(self.ga_results, 'fitness_overall')
        ga_fitness_epoch_history = self._get_history_data(self.ga_results, 'fitness_epoch')
        ga_avg_fitness_epoch_history = self._get_history_data(self.ga_results, 'avg_fitness_epoch')
        
        if ga_fitness_overall_history:
            epochs_ga = list(range(len(ga_fitness_overall_history)))
            fig.add_trace(go.Scatter(x=epochs_ga, y=ga_fitness_overall_history,
                                     mode='lines+markers', name='Melhor Fitness Global (GA)',
                                     line=dict(color='blue', width=2)))
            if ga_fitness_epoch_history and len(ga_fitness_epoch_history) == len(epochs_ga):
                 fig.add_trace(go.Scatter(x=epochs_ga, y=ga_fitness_epoch_history, mode='lines', 
                                          name='Melhor Fitness da Geração (GA)', 
                                          line=dict(color='cyan', width=1, dash='dot')))
            if ga_avg_fitness_epoch_history and len(ga_avg_fitness_epoch_history) == len(epochs_ga):
                 fig.add_trace(go.Scatter(x=epochs_ga, y=ga_avg_fitness_epoch_history, mode='lines', 
                                          name='Fitness Médio da Geração (GA)', 
                                          line=dict(color='lightskyblue', width=1, dash='dash')))

        # PSO Data
        pso_fitness_overall_history = self._get_history_data(self.pso_results, 'fitness_overall')
        pso_fitness_epoch_history = self._get_history_data(self.pso_results, 'fitness_epoch')
        pso_avg_fitness_epoch_history = self._get_history_data(self.pso_results, 'avg_fitness_epoch')

        if pso_fitness_overall_history:
            epochs_pso = list(range(len(pso_fitness_overall_history)))
            fig.add_trace(go.Scatter(x=epochs_pso, y=pso_fitness_overall_history,
                                     mode='lines+markers', name='Melhor Fitness Global (PSO)',
                                     line=dict(color='red', width=2)))
            if pso_fitness_epoch_history and len(pso_fitness_epoch_history) == len(epochs_pso):
                 fig.add_trace(go.Scatter(x=epochs_pso, y=pso_fitness_epoch_history, mode='lines', 
                                          name='Melhor Fitness da Iteração (PSO)', 
                                          line=dict(color='magenta', width=1, dash='dot')))
            if pso_avg_fitness_epoch_history and len(pso_avg_fitness_epoch_history) == len(epochs_pso):
                 fig.add_trace(go.Scatter(x=epochs_pso, y=pso_avg_fitness_epoch_history, mode='lines', 
                                          name='Fitness Médio da Iteração (PSO)', 
                                          line=dict(color='lightcoral', width=1, dash='dash')))
        
        if not fig.data:
            st.write("Nenhum dado de convergência para plotar. Verifique se os algoritmos de otimização registraram o histórico de fitness corretamente.")
        else:
            fig.update_layout(
                xaxis_title="Geração / Iteração",
                yaxis_title="Valor do Fitness (Maior é Melhor, ex: Acurácia)",
                legend_title_text='Métricas de Fitness',
                yaxis_range=[0, 1.05]
            )
            st.plotly_chart(fig, use_container_width=True)

    def plot_selected_features_count_over_time(self):
        st.subheader("Número de Features Numéricas Selecionadas ao Longo das Gerações/Iterações")
        fig = go.Figure()

        ga_features_history = self._get_history_data(self.ga_results, 'features_count')
        if ga_features_history:
            generations_ga = list(range(len(ga_features_history)))
            fig.add_trace(go.Scatter(x=generations_ga, y=ga_features_history,
                                     mode='lines+markers', name='Features Selecionadas (GA)'))
            # print(f"DEBUG Viz: Plot de contagem de features GA, comprimento: {len(ga_features_history)}")

        pso_features_history = self._get_history_data(self.pso_results, 'features_count')
        if pso_features_history:
            iterations_pso = list(range(len(pso_features_history)))
            fig.add_trace(go.Scatter(x=iterations_pso, y=pso_features_history,
                                     mode='lines+markers', name='Features Selecionadas (PSO)'))
            # print(f"DEBUG Viz: Plot de contagem de features PSO, comprimento: {len(pso_features_history)}")
        
        if self.total_numerical_features_original > 0:
            max_gens_iters = 0
            if ga_features_history: max_gens_iters = max(max_gens_iters, len(ga_features_history))
            if pso_features_history: max_gens_iters = max(max_gens_iters, len(pso_features_history))
            
            if max_gens_iters > 0:
                fig.add_trace(go.Scatter(x=list(range(max_gens_iters)), 
                                         y=[self.total_numerical_features_original] * max_gens_iters,
                                         mode='lines', name='Total de Features Numéricas Originais', 
                                         line=dict(dash='dot', color='grey')))

        if not fig.data:
            st.write("Nenhum dado de contagem de features selecionadas para plotar.")
        else:
            fig.update_layout(xaxis_title="Geração / Iteração", yaxis_title="Número de Features Numéricas Selecionadas")
            st.plotly_chart(fig, use_container_width=True)

    def display_performance_metrics(self):
        st.subheader("Métricas Finais de Desempenho")
        data = []
        reduction_ga_text = "N/A"
        reduction_pso_text = "N/A"

        if self.ga_results:
            num_selected_ga = self.ga_results.get('selected_features_count', 0)
            if self.total_numerical_features_original > 0 and isinstance(num_selected_ga, int):
                reduction_ga = ((self.total_numerical_features_original - num_selected_ga) / self.total_numerical_features_original) * 100
                reduction_ga_text = f"{reduction_ga:.1f}% ({self.total_numerical_features_original - num_selected_ga} features reduzidas)"
            elif self.total_numerical_features_original == 0:
                reduction_ga_text = "N/A (sem features numéricas)"
            else:
                reduction_ga_text = "Erro ao calcular redução GA"
            
            ga_hist_len = len(self._get_history_data(self.ga_results, 'fitness_overall'))
            data.append({
                "Algoritmo": "GA",
                "Acurácia Final (Features Selecionadas)": f"{self.ga_results.get('best_fitness', -1.0):.4f}",
                "Features Numéricas Selecionadas": num_selected_ga,
                "Total de Gerações Executadas": ga_hist_len -1 if ga_hist_len > 0 else 0
            })
        
        if self.pso_results:
            num_selected_pso = self.pso_results.get('selected_features_count', 0)
            if self.total_numerical_features_original > 0 and isinstance(num_selected_pso, int):
                reduction_pso = ((self.total_numerical_features_original - num_selected_pso) / self.total_numerical_features_original) * 100
                reduction_pso_text = f"{reduction_pso:.1f}% ({self.total_numerical_features_original - num_selected_pso} features reduzidas)"
            elif self.total_numerical_features_original == 0:
                reduction_pso_text = "N/A (sem features numéricas)"
            else:
                reduction_pso_text = "Erro ao calcular redução PSO"

            pso_hist_len = len(self._get_history_data(self.pso_results, 'fitness_overall'))
            data.append({
                "Algoritmo": "PSO",
                "Acurácia Final (Features Selecionadas)": f"{self.pso_results.get('best_fitness', -1.0):.4f}",
                "Features Numéricas Selecionadas": num_selected_pso,
                "Total de Iterações Executadas": pso_hist_len -1 if pso_hist_len > 0 else 0
            })
        
        if data:
            df_metrics = pd.DataFrame(data)
            st.table(df_metrics.set_index("Algoritmo"))

            st.markdown("---")
            st.subheader("Acurácia Final Detalhada")
            
            num_cols = 0
            if self.ga_results: num_cols +=1
            if self.pso_results: num_cols +=1
            
            if num_cols > 0:
                cols_accuracy = st.columns(num_cols)
                current_col = 0
                if self.ga_results:
                    with cols_accuracy[current_col]:
                        st.metric(label="Acurácia Final GA (Features Selecionadas)", 
                                  value=f"{self.ga_results.get('best_fitness', -1.0):.4f}")
                    current_col+=1
                
                if self.pso_results:
                    with cols_accuracy[current_col]:
                        st.metric(label="Acurácia Final PSO (Features Selecionadas)", 
                                  value=f"{self.pso_results.get('best_fitness', -1.0):.4f}")
            
            st.markdown("---") 
            st.subheader("Redução de Complexidade (Features Numéricas)")
            if num_cols > 0:
                cols_reduction = st.columns(num_cols)
                current_col = 0
                if self.ga_results:
                    with cols_reduction[current_col]:
                        st.metric(label="Redução GA", value=reduction_ga_text)
                    current_col+=1
                if self.pso_results:
                    with cols_reduction[current_col]:
                        st.metric(label="Redução PSO", value=reduction_pso_text)
            else:
                 st.write("Nenhuma métrica de redução para exibir.")

        else:
            st.write("Nenhuma métrica de desempenho para exibir.")
        
    def display_feature_selection_details(self):
        st.subheader("Detalhes das Features Numéricas Selecionadas (Solução Final)")
        
        # GA
        if self.ga_results and 'best_solution_vector' in self.ga_results:
            st.markdown("#### Algoritmo Genético (GA)")
            ga_mask = np.array(self.ga_results['best_solution_vector'])
            st.markdown(f"**Máscara Binária:** `{ga_mask.astype(int)}`")
            
            ga_selected_indices = self.ga_results.get('selected_features_indices')
            if ga_selected_indices is not None:
                st.markdown(f"**Índices Selecionados:** `{ga_selected_indices}`")
                if self.numerical_feature_names and self.total_numerical_features_original == len(ga_mask):
                    selected_names = [self.numerical_feature_names[i] for i in ga_selected_indices if i < len(self.numerical_feature_names)]
                    if selected_names:
                        st.markdown("**Nomes das Features Selecionadas:**")
                        st.text(", ".join(selected_names))
                    else:
                        st.markdown("*Nenhuma feature numérica selecionada com nome correspondente ou lista de nomes vazia.*")
                elif not self.numerical_feature_names:
                     st.markdown("*Nomes das features numéricas originais não fornecidos para exibição.*")
            else:
                st.markdown("*Índices de features selecionadas não disponíveis no resultado do GA.*")
            st.markdown("---")


        # PSO
        if self.pso_results and 'best_solution_vector' in self.pso_results:
            st.markdown("#### Otimização por Enxame de Partículas (PSO)")
            pso_mask = np.array(self.pso_results['best_solution_vector'])
            st.markdown(f"**Máscara Binária:** `{pso_mask.astype(int)}`")

            pso_selected_indices = self.pso_results.get('selected_features_indices')
            if pso_selected_indices is not None:
                st.markdown(f"**Índices Selecionados:** `{pso_selected_indices}`")
                if self.numerical_feature_names and self.total_numerical_features_original == len(pso_mask):
                    selected_names = [self.numerical_feature_names[i] for i in pso_selected_indices if i < len(self.numerical_feature_names)]
                    if selected_names:
                        st.markdown("**Nomes das Features Selecionadas:**")
                        st.text(", ".join(selected_names))
                    else:
                        st.markdown("*Nenhuma feature numérica selecionada com nome correspondente ou lista de nomes vazia.*")
                elif not self.numerical_feature_names:
                     st.markdown("*Nomes das features numéricas originais não fornecidos para exibição.*")
            else:
                st.markdown("*Índices de features selecionadas não disponíveis no resultado do PSO.*")
            st.markdown("---")
        
        if not (self.ga_results and 'best_solution_vector' in self.ga_results) and \
           not (self.pso_results and 'best_solution_vector' in self.pso_results):
             st.write("Nenhum detalhe de seleção de features para exibir.")

        # Tabela com todos os dados coletados
        st.markdown("---")
        st.subheader("Tabela Completa de Dados Coletados")

        # Criar DataFrame com todos os dados históricos
        all_data = []

        if self.ga_results and 'history' in self.ga_results:
            ga_history = self.ga_results['history']
            max_len_ga = max([len(ga_history.get(key, [])) for key in ga_history.keys()] + [0])
            
            for i in range(max_len_ga):
                row = {
                    'Algoritmo': 'GA',
                    'Geração/Iteração': i,
                    'Fitness Global': ga_history.get('fitness_overall', [])[i] if i < len(ga_history.get('fitness_overall', [])) else None,
                    'Fitness da Época': ga_history.get('fitness_epoch', [])[i] if i < len(ga_history.get('fitness_epoch', [])) else None,
                    'Fitness Médio da Época': ga_history.get('avg_fitness_epoch', [])[i] if i < len(ga_history.get('avg_fitness_epoch', [])) else None,
                    'Features Selecionadas': ga_history.get('features_count', [])[i] if i < len(ga_history.get('features_count', [])) else None
                }
                all_data.append(row)

        if self.pso_results and 'history' in self.pso_results:
            pso_history = self.pso_results['history']
            max_len_pso = max([len(pso_history.get(key, [])) for key in pso_history.keys()] + [0])
            
            for i in range(max_len_pso):
                row = {
                    'Algoritmo': 'PSO',
                    'Geração/Iteração': i,
                    'Fitness Global': pso_history.get('fitness_overall', [])[i] if i < len(pso_history.get('fitness_overall', [])) else None,
                    'Fitness da Época': pso_history.get('fitness_epoch', [])[i] if i < len(pso_history.get('fitness_epoch', [])) else None,
                    'Fitness Médio da Época': pso_history.get('avg_fitness_epoch', [])[i] if i < len(pso_history.get('avg_fitness_epoch', [])) else None,
                    'Features Selecionadas': pso_history.get('features_count', [])[i] if i < len(pso_history.get('features_count', [])) else None
                }
                all_data.append(row)

        if all_data:
            df_all_data = pd.DataFrame(all_data)
            
            # Filtrar dados por algoritmo
            algorithm_filter = st.selectbox("Filtrar por Algoritmo:", ["Todos", "GA", "PSO"])
            
            if algorithm_filter != "Todos":
                df_filtered = df_all_data[df_all_data['Algoritmo'] == algorithm_filter]
            else:
                df_filtered = df_all_data
            
            # Mostrar estatísticas básicas
            if not df_filtered.empty:
                st.markdown("**Estatísticas dos Dados:**")
                cols_stats = st.columns(3)
                with cols_stats[0]:
                    st.metric("Total de Registros", len(df_filtered))
                with cols_stats[1]:
                    if 'Fitness Global' in df_filtered.columns:
                        max_fitness = df_filtered['Fitness Global'].max()
                        st.metric("Melhor Fitness Global", f"{max_fitness:.4f}" if pd.notna(max_fitness) else "N/A")
                with cols_stats[2]:
                    if 'Features Selecionadas' in df_filtered.columns:
                        avg_features = df_filtered['Features Selecionadas'].mean()
                        st.metric("Média de Features", f"{avg_features:.1f}" if pd.notna(avg_features) else "N/A")
                
                # Exibir tabela
                st.dataframe(df_filtered, use_container_width=True, height=400)
                
                # Opção para download
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    label="📥 Baixar dados como CSV",
                    data=csv,
                    file_name=f"dados_execucao_{algorithm_filter.lower()}.csv",
                    mime="text/csv"
                )
            else:
                st.write("Nenhum dado disponível para o filtro selecionado.")
        else:
            st.write("Nenhum dado histórico disponível para exibir.")