# # from typing import Any, Dict, List, Optional

# # import matplotlib.pyplot as plt
# # import numpy as np
# # import pandas as pd  # type: ignore
# # import plotly.express as px  # type: ignore
# # import plotly.graph_objects as go  # type: ignore
# # import streamlit as st
# # from matplotlib.ticker import MaxNLocator


# # class ResultsVisualizationPage:
# #     def __init__(self,
# #                  ga_results: Optional[Dict[str, Any]] = None,
# #                  pso_results: Optional[Dict[str, Any]] = None,
# #                 #  hybrid_results: Optional[Dict[str, Any]] = None,
# #                  feature_names: Optional[List[str]] = None):
# #         """
# #         Página para visualizar e comparar resultados dos três algoritmos de seleção de características.

# #         Parâmetros:
# #         -----------
# #         ga_results : dict
# #             Resultados do algoritmo genético
# #         pso_results : dict
# #             Resultados do algoritmo PSO
# #         hybrid_results : dict
# #             Resultados do algoritmo híbrido PSO-GA
# #         feature_names : list
# #             Lista com os nomes das características
# #         """
# #         st.title("Visualização Comparativa de Resultados")

# #         # Se não houver resultados, exibir mensagem
# #         if ga_results is None and pso_results is None:
# #             st.warning(
# #                 "Nenhum resultado para mostrar. Execute os algoritmos primeiro.")
# #             return

# #         # Determinar quais modelos estão disponíveis
# #         available_models = []
# #         if ga_results is not None:
# #             available_models.append("GA")
# #         if pso_results is not None:
# #             available_models.append("PSO")

# #         st.write(
# #             f"Comparando resultados de {len(available_models)} modelos: {', '.join(available_models)}")

# #         # Criar abas para diferentes tipos de visualizações
# #         tab_convergence, tab_features, tab_performance, tab_selection = st.tabs([
# #             "Convergência", "Features Selecionadas", "Métricas de Desempenho", "Seleção de Features"
# #         ])

# #         # Tab 1: Convergência de Fitness
# #         with tab_convergence:
# #             self.plot_fitness_convergence(
# #                 ga_results, pso_results)

# #         # Tab 2: Número de Features Selecionadas
# #         with tab_features:
# #             self.plot_feature_count(ga_results, pso_results)

# #         # Tab 3: Métricas de Desempenho
# #         with tab_performance:
# #             self.plot_performance_comparison(
# #                 ga_results, pso_results)

# #         # Tab 4: Seleção de Features
# #         with tab_selection:
# #             self.plot_feature_selection(
# #                 ga_results, pso_results, feature_names)
        
# #         # # Botão para limpar resultados
# #         # if st.button("Limpar Todos os Resultados"):
# #         #     for key in ['ga_results', 'pso_results', 'hybrid_results']:
# #         #         if key in st.session_state:
# #         #             st.session_state[key] = None
# #         #     st.experimental_rerun()

# #     def plot_fitness_convergence(self, ga_results, pso_results):
# #         """Plotar gráfico de convergência de fitness para todos os modelos disponíveis."""
# #         st.subheader("Convergência do Fitness")

# #         # Usar Plotly para criar gráfico interativo
# #         fig = go.Figure()

# #         # Adicionar dados ao gráfico para cada modelo disponível
# #         if ga_results and 'best_fitness_history' in ga_results:
# #             fig.add_trace(go.Scatter(
# #                 x=list(range(len(ga_results['best_fitness_history']))),
# #                 y=ga_results['best_fitness_history'],
# #                 mode='lines',
# #                 name='Melhor Fitness (GA)',
# #                 line=dict(color='blue', width=2)
# #             ))
# #             fig.add_trace(go.Scatter(
# #                 x=list(range(len(ga_results['avg_fitness_history']))),
# #                 y=ga_results['avg_fitness_history'],
# #                 mode='lines',
# #                 name='Fitness Médio (GA)',
# #                 line=dict(color='blue', width=1, dash='dash')
# #             ))

# #         if pso_results and 'best_fitness_history' in pso_results:
# #             fig.add_trace(go.Scatter(
# #                 x=list(range(len(pso_results['best_fitness_history']))),
# #                 y=pso_results['best_fitness_history'],
# #                 mode='lines',
# #                 name='Melhor Fitness (PSO)',
# #                 line=dict(color='red', width=2)
# #             ))
# #             fig.add_trace(go.Scatter(
# #                 x=list(range(len(pso_results['avg_fitness_history']))),
# #                 y=pso_results['avg_fitness_history'],
# #                 mode='lines',
# #                 name='Fitness Médio (PSO)',
# #                 line=dict(color='red', width=1, dash='dash')
# #             ))

# #         # if hybrid_results and 'best_fitness_history' in hybrid_results:
# #         #     fig.add_trace(go.Scatter(
# #         #         x=list(range(len(hybrid_results['best_fitness_history']))),
# #         #         y=hybrid_results['best_fitness_history'],
# #         #         mode='lines',
# #         #         name='Melhor Fitness (Híbrido)',
# #         #         line=dict(color='green', width=2)
# #         #     ))
# #         #     fig.add_trace(go.Scatter(
# #         #         x=list(range(len(hybrid_results['avg_fitness_history']))),
# #         #         y=hybrid_results['avg_fitness_history'],
# #         #         mode='lines',
# #         #         name='Fitness Médio (Híbrido)',
# #         #         line=dict(color='green', width=1, dash='dash')
# #         #     ))

# #         # Configurar layout do gráfico
# #         fig.update_layout(
# #             title='Convergência do Fitness ao Longo das Gerações/Iterações',
# #             xaxis_title='Geração/Iteração',
# #             yaxis_title='Valor de Fitness',
# #             hovermode='x unified',
# #             legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
# #         )

# #         # Exibir gráfico interativo no Streamlit
# #         st.plotly_chart(fig, use_container_width=True)

# #         # Adicionar métricas comparativas
# #         st.subheader("Métricas de Convergência")
# #         col1, col2, col3 = st.columns(3)

# #         with col1:
# #             if ga_results and 'best_fitness_history' in ga_results:
# #                 st.metric("Melhor Fitness (GA)",
# #                           f"{max(ga_results['best_fitness_history']):.4f}")

# #         with col2:
# #             if pso_results and 'best_fitness_history' in pso_results:
# #                 st.metric("Melhor Fitness (PSO)",
# #                           f"{max(pso_results['best_fitness_history']):.4f}")

# #         # with col3:
# #         #     if hybrid_results and 'best_fitness_history' in hybrid_results:
# #         #         st.metric("Melhor Fitness (Híbrido)",
# #         #                   f"{max(hybrid_results['best_fitness_history']):.4f}")

# #     def plot_feature_count(self, ga_results, pso_results):
# #         """Plotar número de features selecionadas ao longo das gerações para todos os modelos."""
# #         st.subheader("Número de Features Selecionadas")

# #         # Verificar se há dados disponíveis
# #         if not any(x and 'feature_count_history' in x for x in [ga_results, pso_results]):
# #             st.warning("Não há dados sobre contagem de features disponíveis.")
# #             return

# #         # Usar Plotly para criar gráfico interativo
# #         fig = go.Figure()

# #         # Determinar o número máximo de gerações/iterações
# #         max_generations = 0
# #         if ga_results and 'feature_count_history' in ga_results:
# #             max_generations = max(max_generations, len(ga_results['feature_count_history']))
# #         if pso_results and 'feature_count_history' in pso_results:
# #             max_generations = max(max_generations, len(pso_results['feature_count_history']))
# #         # if hybrid_results and 'feature_count_history' in hybrid_results:
# #         #     max_generations = max(max_generations, len(hybrid_results['feature_count_history']))
        
# #         # Adicionar um pequeno buffer para visualização
# #         x_max = max_generations * 1.1 if max_generations > 0 else 10

# #         # Número total de features (assumindo que é o mesmo para todos os modelos)
# #         n_features = None
# #         if ga_results and 'n_features' in ga_results:
# #             n_features = ga_results['n_features']
# #         elif pso_results and 'n_features' in pso_results:
# #             n_features = pso_results['n_features']
# #         # elif hybrid_results and 'n_features' in hybrid_results:
# #         #     n_features = hybrid_results['n_features']

# #         # Adicionar linha de total de features se disponível
# #         if n_features is not None:
# #             fig.add_trace(go.Scatter(
# #                 x=[0, x_max],  # Use o valor máximo calculado
# #                 y=[n_features, n_features],
# #                 mode='lines',
# #                 name=f'Total de Features ({n_features})',
# #                 line=dict(color='gray', width=1, dash='dash')
# #             ))

# #         # Adicionar dados para cada modelo disponível
# #         if ga_results and 'feature_count_history' in ga_results:
# #             final_count_ga = ga_results['feature_count_history'][-1]
# #             fig.add_trace(go.Scatter(
# #                 x=list(range(len(ga_results['feature_count_history']))),
# #                 y=ga_results['feature_count_history'],
# #                 mode='lines',
# #                 name=f'GA ({final_count_ga} features)',
# #                 line=dict(color='blue', width=2)
# #             ))

# #         if pso_results and 'feature_count_history' in pso_results:
# #             final_count_pso = pso_results['feature_count_history'][-1]
# #             fig.add_trace(go.Scatter(
# #                 x=list(range(len(pso_results['feature_count_history']))),
# #                 y=pso_results['feature_count_history'],
# #                 mode='lines',
# #                 name=f'PSO ({final_count_pso} features)',
# #                 line=dict(color='red', width=2)
# #             ))

# #         # if hybrid_results and 'feature_count_history' in hybrid_results:
# #         #     final_count_hybrid = hybrid_results['feature_count_history'][-1]
# #         #     fig.add_trace(go.Scatter(
# #         #         x=list(range(len(hybrid_results['feature_count_history']))),
# #         #         y=hybrid_results['feature_count_history'],
# #         #         mode='lines',
# #         #         name=f'Híbrido ({final_count_hybrid} features)',
# #         #         line=dict(color='green', width=2)
# #             # ))

# #         # Configurar layout
# #         fig.update_layout(
# #             title='Número de Features Selecionadas ao Longo das Gerações/Iterações',
# #             xaxis_title='Geração/Iteração',
# #             yaxis_title='Número de Features',
# #             hovermode='x unified',
# #             legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
# #             xaxis=dict(range=[0, x_max])  # Definir limites do eixo x explicitamente

# #         )

# #         # Garantir que o eixo y mostre apenas valores inteiros
# #         fig.update_yaxes(tick0=0, dtick=1)

# #         # Exibir gráfico
# #         st.plotly_chart(fig, use_container_width=True)

# #         # Mostrar percentuais de redução
# #         if n_features:
# #             st.subheader("Redução de Complexidade")
# #             cols = st.columns(3)

# #             with cols[0]:
# #                 if ga_results and 'feature_count_history' in ga_results:
# #                     final_count = ga_results['feature_count_history'][-1]
# #                     reduction = (1 - final_count/n_features) * 100
# #                     st.metric("Redução (GA)", f"{reduction:.1f}%",
# #                               delta=f"-{n_features - final_count} features")

# #             with cols[1]:
# #                 if pso_results and 'feature_count_history' in pso_results:
# #                     final_count = pso_results['feature_count_history'][-1]
# #                     reduction = (1 - final_count/n_features) * 100
# #                     st.metric("Redução (PSO)", f"{reduction:.1f}%",
# #                               delta=f"-{n_features - final_count} features")

# #             # with cols[2]:
# #             #     if hybrid_results and 'feature_count_history' in hybrid_results:
# #             #         final_count = hybrid_results['feature_count_history'][-1]
# #             #         reduction = (1 - final_count/n_features) * 100
# #             #         st.metric("Redução (Híbrido)", f"{reduction:.1f}%",
# #             #                   delta=f"-{n_features - final_count} features")

# #     def plot_performance_comparison(self, ga_results, pso_results):
# #         """Plotar comparação de desempenho entre os modelos."""
# #         st.subheader("Comparação de Desempenho")

# #         # Criar DataFrame para comparação
# #         data = []
# #         if ga_results:
# #             data.append({
# #                 'Modelo': 'GA',
# #                 'Acurácia': ga_results.get('best_fitness', 0),
# #                 'Tempo de Execução (s)': sum(ga_results.get('execution_times', [0])),
# #                 'Features Selecionadas': ga_results.get('feature_count_history', [0])[-1] if 'feature_count_history' in ga_results else 0
# #             })

# #         if pso_results:
# #             data.append({
# #                 'Modelo': 'PSO',
# #                 'Acurácia': pso_results.get('best_fitness', 0),
# #                 'Tempo de Execução (s)': sum(pso_results.get('execution_times', [0])),
# #                 'Features Selecionadas': pso_results.get('feature_count_history', [0])[-1] if 'feature_count_history' in pso_results else 0
# #             })

# #         # if hybrid_results:
# #         #     data.append({
# #         #         'Modelo': 'Híbrido PSO-GA',
# #         #         'Acurácia': hybrid_results.get('best_fitness', 0),
# #         #         'Tempo de Execução (s)': sum(hybrid_results.get('execution_times', [0])),
# #         #         'Features Selecionadas': hybrid_results.get('feature_count_history', [0])[-1] if 'feature_count_history' in hybrid_results else 0
# #         #     })

# #         if not data:
# #             st.warning("Não há dados de desempenho disponíveis.")
# #             return

# #         df = pd.DataFrame(data)

# #         # Exibir tabela
# #         st.dataframe(df.style.highlight_max(subset=['Acurácia']).highlight_min(
# #             subset=['Tempo de Execução (s)', 'Features Selecionadas']))

# #         # Gráfico de barras para acurácia
# #         fig_accuracy = px.bar(
# #             df,
# #             x='Modelo',
# #             y='Acurácia',
# #             color='Modelo',
# #             text_auto='.4f',
# #             title='Acurácia por Modelo',
# #             height=400
# #         )
# #         st.plotly_chart(fig_accuracy, use_container_width=True)

# #         # Gráfico de barras para tempo de execução
# #         fig_time = px.bar(
# #             df,
# #             x='Modelo',
# #             y='Tempo de Execução (s)',
# #             color='Modelo',
# #             text_auto='.2f',
# #             title='Tempo de Execução por Modelo',
# #             height=400
# #         )
# #         st.plotly_chart(fig_time, use_container_width=True)

# #         # Radar chart para comparação multidimensional
# #         # Normalizar os dados para o radar chart
# #         df_radar = df.copy()
# #         df_radar['Acurácia_norm'] = df_radar['Acurácia'] / \
# #             df_radar['Acurácia'].max()
# #         # Invertido para que menor tempo = melhor
# #         df_radar['Tempo_norm'] = 1 - \
# #             (df_radar['Tempo de Execução (s)'] /
# #              df_radar['Tempo de Execução (s)'].max())
# #         # Invertido para que menos features = melhor
# #         df_radar['Features_norm'] = 1 - \
# #             (df_radar['Features Selecionadas'] /
# #              df_radar['Features Selecionadas'].max())

# #         fig_radar = go.Figure()

# #         for i, row in df_radar.iterrows():
# #             fig_radar.add_trace(go.Scatterpolar(
# #                 r=[row['Acurácia_norm'], row['Tempo_norm'], row['Features_norm']],
# #                 theta=['Acurácia', 'Eficiência de Tempo',
# #                        'Redução de Features'],
# #                 fill='toself',
# #                 name=row['Modelo']
# #             ))

# #         fig_radar.update_layout(
# #             polar=dict(
# #                 radialaxis=dict(
# #                     visible=True,
# #                     range=[0, 1]
# #                 )),
# #             showlegend=True,
# #             title='Comparação Multidimensional de Modelos'
# #         )

# #         st.plotly_chart(fig_radar, use_container_width=True)

# #     def plot_feature_selection(self, ga_results, pso_results, feature_names):
# #         """Visualizar quais features foram selecionadas por cada modelo."""
# #         st.subheader("Features Selecionadas por Modelo")

# #         # Verificar se temos os dados necessários
# #         if not any(x and 'best_solution' in x for x in [ga_results, pso_results]):
# #             st.warning("Não há dados sobre features selecionadas disponíveis.")
# #             return

# #         # Se não temos nomes, usar índices
# #         if feature_names is None:
# #             n_features = 0
# #             if ga_results and 'best_solution' in ga_results:
# #                 n_features = len(ga_results['best_solution'])
# #             elif pso_results and 'best_solution' in pso_results:
# #                 n_features = len(pso_results['best_solution'])
# #             # elif hybrid_results and 'best_solution' in hybrid_results:
# #             #     n_features = len(hybrid_results['best_solution'])

# #             feature_names = [f"Feature {i+1}" for i in range(n_features)]

# #         # Criar DataFrame para visualização
# #         feature_selection = pd.DataFrame({'Feature': feature_names})

# #         # Adicionar seleções de cada modelo
# #         if ga_results and 'best_solution' in ga_results:
# #             feature_selection['GA'] = ga_results['best_solution']

# #         if pso_results and 'best_solution' in pso_results:
# #             feature_selection['PSO'] = [1 if x else 0 for x in pso_results['best_solution']]

# #         # if hybrid_results and 'best_solution' in hybrid_results:
# #         #     feature_selection['Híbrido'] = hybrid_results['best_solution']

# #         # Calcular quantos modelos selecionaram cada feature
# #         feature_selection['Total'] = feature_selection.iloc[:, 1:].sum(axis=1)

# #         # Ordenar pelo número de modelos que selecionaram cada feature
# #         feature_selection = feature_selection.sort_values(
# #             by='Total', ascending=False)

# #         # Heatmap de seleção de features
# #         fig_heatmap = px.imshow(
# #             feature_selection.iloc[:, 1:-1].transpose(),
# #             x=feature_selection['Feature'],
# #             y=feature_selection.columns[1:-1],
# #             color_continuous_scale=[[0, 'white'], [1, 'green']],
# #             labels=dict(x="Feature", y="Modelo", color="Selecionada"),
# #             title="Heatmap de Seleção de Features por Modelo"
# #         )

# #         fig_heatmap.update_layout(
# #             xaxis_tickangle=45,
# #             yaxis_title="Modelo",
# #             xaxis_title="Feature"
# #         )

# #         # Exibir heatmap
# #         st.plotly_chart(fig_heatmap, use_container_width=True)

# #         # Mostrar tabela com as features selecionadas
# #         st.subheader("Features Selecionadas (Detalhado)")
# #         st.dataframe(feature_selection.style.background_gradient(
# #             subset=['Total'], cmap='Greens'), height=400)

# #         # Diagrama de Venn para visualizar intersecções (usando gráficos de barras agrupados)
# #         st.subheader("Concordância entre Modelos")

# #         # Criar DataFrame para contagem de concordância
# #         concordance_data = []

# #         # Features selecionadas por cada modelo
# #         ga_features = set() if 'GA' not in feature_selection.columns else set(
# #             feature_selection[feature_selection['GA'] == 1]['Feature'])
# #         pso_features = set() if 'PSO' not in feature_selection.columns else set(
# #             feature_selection[feature_selection['PSO'] == 1]['Feature'])
# #         # hybrid_features = set() if 'Híbrido' not in feature_selection.columns else set(
# #         #     feature_selection[feature_selection['Híbrido'] == 1]['Feature'])

# #         # Calcular intersecções
# #         if 'GA' in feature_selection.columns and 'PSO' in feature_selection.columns:
# #             concordance_data.append({
# #                 'Comparação': 'GA ∩ PSO',
# #                 'Quantidade': len(ga_features.intersection(pso_features)),
# #                 'Porcentagem GA': len(ga_features.intersection(pso_features)) / len(ga_features) * 100 if ga_features else 0,
# #                 'Porcentagem PSO': len(ga_features.intersection(pso_features)) / len(pso_features) * 100 if pso_features else 0
# #             })

# #         # if 'GA' in feature_selection.columns and 'Híbrido' in feature_selection.columns:
# #         #     concordance_data.append({
# #         #         'Comparação': 'GA ∩ Híbrido',
# #         #         'Quantidade': len(ga_features.intersection(hybrid_features)),
# #         #         'Porcentagem GA': len(ga_features.intersection(hybrid_features)) / len(ga_features) * 100 if ga_features else 0,
# #         #         'Porcentagem Híbrido': len(ga_features.intersection(hybrid_features)) / len(hybrid_features) * 100 if hybrid_features else 0
# #         #     })

# #         # if 'PSO' in feature_selection.columns and 'Híbrido' in feature_selection.columns:
# #         #     concordance_data.append({
# #         #         'Comparação': 'PSO ∩ Híbrido',
# #         #         'Quantidade': len(pso_features.intersection(hybrid_features)),
# #         #         'Porcentagem PSO': len(pso_features.intersection(hybrid_features)) / len(pso_features) * 100 if pso_features else 0,
# #         #         'Porcentagem Híbrido': len(pso_features.intersection(hybrid_features)) / len(hybrid_features) * 100 if hybrid_features else 0
# #         #     })

# #         # if 'GA' in feature_selection.columns and 'PSO' in feature_selection.columns and 'Híbrido' in feature_selection.columns:
# #         #     concordance_data.append({
# #         #         'Comparação': 'GA ∩ PSO ∩ Híbrido',
# #         #         'Quantidade': len(ga_features.intersection(pso_features).intersection(hybrid_features)),
# #         #         'Porcentagem GA': len(ga_features.intersection(pso_features).intersection(hybrid_features)) / len(ga_features) * 100 if ga_features else 0,
# #         #         'Porcentagem PSO': len(ga_features.intersection(pso_features).intersection(hybrid_features)) / len(pso_features) * 100 if pso_features else 0,
# #         #         'Porcentagem Híbrido': len(ga_features.intersection(pso_features).intersection(hybrid_features)) / len(hybrid_features) * 100 if hybrid_features else 0
# #         #     })

# #         if concordance_data:
# #             df_concordance = pd.DataFrame(concordance_data)

# #             # Mostrar dados de concordância
# #             st.dataframe(df_concordance)

# #             # Gráfico de barras para visualização de concordância
# #             fig_concordance = px.bar(
# #                 df_concordance,
# #                 x='Comparação',
# #                 y='Quantidade',
# #                 text='Quantidade',
# #                 title='Número de Features em Comum entre Modelos',
# #                 height=400
# #             )

# #             st.plotly_chart(fig_concordance, use_container_width=True)
# import streamlit as st
# import plotly.graph_objects as go
# import numpy as np
# import pandas as pd

# class ResultsVisualizationPage:
#     def __init__(self, ga_results, pso_results, numerical_feature_names=None):
#         self.ga_results = ga_results
#         self.pso_results = pso_results
#         self.numerical_feature_names = numerical_feature_names if numerical_feature_names else []
        
#         # Determinar o número total de features numéricas originais
#         self.total_numerical_features_original = len(self.numerical_feature_names)
#         if self.total_numerical_features_original == 0: # Fallback se nomes não foram passados mas resultados existem
#             if self.ga_results and 'best_solution_vector' in self.ga_results:
#                 self.total_numerical_features_original = len(self.ga_results['best_solution_vector'])
#             elif self.pso_results and 'best_solution_vector' in self.pso_results:
#                 self.total_numerical_features_original = len(self.pso_results['best_solution_vector'])


#         st.header("Visualização Comparativa de Resultados")
        
#         if not self.ga_results and not self.pso_results:
#             st.warning("Nenhum resultado disponível para visualização. Execute a seleção de features primeiro.")
#             return

#         tabs_list = ["Convergência", "Contagem de Features Selecionadas", "Métricas de Desempenho", "Detalhes da Seleção de Features"]
        
#         tabs = st.tabs(tabs_list)

#         with tabs[0]:
#             self.plot_convergence()
#         with tabs[1]:
#             self.plot_selected_features_count_over_time()
#         with tabs[2]:
#             self.display_performance_metrics()
#         with tabs[3]:
#             self.display_feature_selection_details()
            
#     def _get_history_data(self, results, key):
#         """Helper para obter dados de histórico com segurança."""
#         if results and 'history' in results and results['history'] and key in results['history']:
#             return results['history'][key]
#         return []

#     # def plot_convergence(self):
#     #     st.subheader("Convergência do Fitness ao Longo das Gerações/Iterações")
#     #     fig = go.Figure()
        
#     #     ga_fitness_history = self._get_history_data(self.ga_results, 'fitness')
#     #     ga_avg_fitness_history = self._get_history_data(self.ga_results, 'avg_fitness')
        
#     #     if ga_fitness_history:
#     #         generations_ga = list(range(len(ga_fitness_history))) 
#     #         fig.add_trace(go.Scatter(x=generations_ga, y=ga_fitness_history,
#     #                                  mode='lines+markers', name='Melhor Fitness (GA)'))
#     #         if ga_avg_fitness_history and len(ga_avg_fitness_history) == len(ga_fitness_history):
#     #             fig.add_trace(go.Scatter(x=generations_ga, y=ga_avg_fitness_history,
#     #                                      mode='lines', name='Fitness Médio (GA)', line=dict(dash='dash')))
#     #         print(f"DEBUG Viz: Plot de fitness GA, comprimento: {len(ga_fitness_history)}")

#     #     pso_fitness_history = self._get_history_data(self.pso_results, 'fitness')
#     #     pso_avg_fitness_history = self._get_history_data(self.pso_results, 'avg_fitness')

#     #     if pso_fitness_history:
#     #         iterations_pso = list(range(len(pso_fitness_history)))
#     #         fig.add_trace(go.Scatter(x=iterations_pso, y=pso_fitness_history,
#     #                                  mode='lines+markers', name='Melhor Fitness (PSO)'))
#     #         if pso_avg_fitness_history and len(pso_avg_fitness_history) == len(pso_fitness_history):
#     #             fig.add_trace(go.Scatter(x=iterations_pso, y=pso_avg_fitness_history,
#     #                                      mode='lines', name='Fitness Médio (PSO)', line=dict(dash='dash')))
#     #         print(f"DEBUG Viz: Plot de fitness PSO, comprimento: {len(pso_fitness_history)}")
        
#     #     if not fig.data:
#     #         st.write("Nenhum dado de convergência para plotar.")
#     #     else:
#     #         fig.update_layout(xaxis_title="Geração / Iteração", yaxis_title="Valor do Fitness (Menor é Melhor)")
#     #         st.plotly_chart(fig, use_container_width=True)

#     # def plot_convergence(self):
#     #     st.subheader("Convergência do Fitness ao Longo das Gerações/Iterações")
#     #     fig = go.Figure()
        
#     #     # GA Data
#     #     ga_fitness_overall_history = self._get_history_data(self.ga_results, 'fitness_overall')
#     #     ga_fitness_epoch_history = self._get_history_data(self.ga_results, 'fitness_epoch')
#     #     ga_avg_fitness_epoch_history = self._get_history_data(self.ga_results, 'avg_fitness_epoch')
        
#     #     if ga_fitness_overall_history:
#     #         epochs_ga = list(range(len(ga_fitness_overall_history)))
#     #         fig.add_trace(go.Scatter(x=epochs_ga, y=ga_fitness_overall_history,
#     #                                  mode='lines+markers', name='Melhor Fitness Global (GA)',
#     #                                  line=dict(color='blue', width=2))) # Azul para GA Global
#     #         if ga_fitness_epoch_history and len(ga_fitness_epoch_history) == len(epochs_ga):
#     #              fig.add_trace(go.Scatter(x=epochs_ga, y=ga_fitness_epoch_history,
#     #                                  mode='lines', name='Melhor Fitness da Geração (GA)',
#     #                                  line=dict(color='cyan', dash='dot'))) # Ciano para GA da Geração
#     #         if ga_avg_fitness_epoch_history and len(ga_avg_fitness_epoch_history) == len(epochs_ga):
#     #             fig.add_trace(go.Scatter(x=epochs_ga, y=ga_avg_fitness_epoch_history,
#     #                                      mode='lines', name='Fitness Médio da Geração (GA)',
#     #                                      line=dict(color='lightblue', dash='dash'))) # Azul claro para GA Médio

#     #     # PSO Data
#     #     pso_fitness_overall_history = self._get_history_data(self.pso_results, 'fitness_overall')
#     #     pso_fitness_epoch_history = self._get_history_data(self.pso_results, 'fitness_epoch')
#     #     pso_avg_fitness_epoch_history = self._get_history_data(self.pso_results, 'avg_fitness_epoch')

#     #     if pso_fitness_overall_history:
#     #         epochs_pso = list(range(len(pso_fitness_overall_history)))
#     #         fig.add_trace(go.Scatter(x=epochs_pso, y=pso_fitness_overall_history,
#     #                                  mode='lines+markers', name='Melhor Fitness Global (PSO)',
#     #                                  line=dict(color='red', width=2))) # Vermelho para PSO Global
#     #         if pso_fitness_epoch_history and len(pso_fitness_epoch_history) == len(epochs_pso):
#     #              fig.add_trace(go.Scatter(x=epochs_pso, y=pso_fitness_epoch_history,
#     #                                  mode='lines', name='Melhor Fitness da Iteração (PSO)',
#     #                                  line=dict(color='magenta', dash='dot'))) # Magenta para PSO da Iteração
#     #         if pso_avg_fitness_epoch_history and len(pso_avg_fitness_epoch_history) == len(epochs_pso):
#     #             fig.add_trace(go.Scatter(x=epochs_pso, y=pso_avg_fitness_epoch_history,
#     #                                      mode='lines', name='Fitness Médio da Iteração (PSO)',
#     #                                      line=dict(color='lightcoral', dash='dash'))) # Coral claro para PSO Médio
        
#     #     if not fig.data:
#     #         st.write("Nenhum dado de convergência para plotar.")
#     #     else:
#     #         fig.update_layout(
#     #             xaxis_title="Geração / Iteração", 
#     #             # ATUALIZAR TÍTULO DO EIXO Y
#     #             yaxis_title="Valor do Fitness (Maior é Melhor, ex: Acurácia)",
#     #             legend_title_text='Métricas de Fitness',
#     #             yaxis_range=[0, 1.05] # Se o fitness for acurácia (0-1)
#     #         )
#     #         st.plotly_chart(fig, use_container_width=True)

#     def plot_convergence(self):
#         st.subheader("Convergência do Fitness ao Longo das Gerações/Iterações")
#         fig = go.Figure()
        
#         # GA Data
#         ga_fitness_overall_history = self._get_history_data(self.ga_results, 'fitness_overall')
#         ga_fitness_epoch_history = self._get_history_data(self.ga_results, 'fitness_epoch')
#         ga_avg_fitness_epoch_history = self._get_history_data(self.ga_results, 'avg_fitness_epoch')
        
#         if ga_fitness_overall_history:
#             epochs_ga = list(range(len(ga_fitness_overall_history)))
#             fig.add_trace(go.Scatter(x=epochs_ga, y=ga_fitness_overall_history,
#                                      mode='lines+markers', name='Melhor Fitness Global (GA)',
#                                      line=dict(color='blue', width=2))) # Azul para GA Global
#             if ga_fitness_epoch_history and len(ga_fitness_epoch_history) == len(epochs_ga):
#                  fig.add_trace(go.Scatter(x=epochs_ga, y=ga_fitness_epoch_history, mode='lines', 
#                                           name='Melhor Fitness da Geração (GA)', 
#                                           line=dict(color='cyan', width=1, dash='dot'))) # Ciano pontilhado para melhor da geração GA
#             if ga_avg_fitness_epoch_history and len(ga_avg_fitness_epoch_history) == len(epochs_ga):
#                  fig.add_trace(go.Scatter(x=epochs_ga, y=ga_avg_fitness_epoch_history, mode='lines', 
#                                           name='Fitness Médio da Geração (GA)', 
#                                           line=dict(color='lightskyblue', width=1, dash='dash'))) # Azul claro tracejado para média da geração GA

#         # PSO Data
#         pso_fitness_overall_history = self._get_history_data(self.pso_results, 'fitness_overall')
#         pso_fitness_epoch_history = self._get_history_data(self.pso_results, 'fitness_epoch')
#         pso_avg_fitness_epoch_history = self._get_history_data(self.pso_results, 'avg_fitness_epoch')

#         if pso_fitness_overall_history:
#             epochs_pso = list(range(len(pso_fitness_overall_history)))
#             fig.add_trace(go.Scatter(x=epochs_pso, y=pso_fitness_overall_history,
#                                      mode='lines+markers', name='Melhor Fitness Global (PSO)',
#                                      line=dict(color='red', width=2))) # Vermelho para PSO Global
#             if pso_fitness_epoch_history and len(pso_fitness_epoch_history) == len(epochs_pso):
#                  fig.add_trace(go.Scatter(x=epochs_pso, y=pso_fitness_epoch_history, mode='lines', 
#                                           name='Melhor Fitness da Iteração (PSO)', 
#                                           line=dict(color='magenta', width=1, dash='dot'))) # Magenta pontilhado para melhor da iteração PSO
#             if pso_avg_fitness_epoch_history and len(pso_avg_fitness_epoch_history) == len(epochs_pso):
#                  fig.add_trace(go.Scatter(x=epochs_pso, y=pso_avg_fitness_epoch_history, mode='lines', 
#                                           name='Fitness Médio da Iteração (PSO)', 
#                                           line=dict(color='lightcoral', width=1, dash='dash'))) # Coral claro tracejado para média da iteração PSO
        
#         if not fig.data:
#             st.write("Nenhum dado de convergência para plotar.")

#     def plot_selected_features_count_over_time(self):
#         st.subheader("Número de Features Numéricas Selecionadas ao Longo das Gerações/Iterações")
#         fig = go.Figure()

#         ga_features_history = self._get_history_data(self.ga_results, 'features_count')
#         if ga_features_history:
#             generations_ga = list(range(len(ga_features_history)))
#             fig.add_trace(go.Scatter(x=generations_ga, y=ga_features_history,
#                                      mode='lines+markers', name='Features Selecionadas (GA)'))
#             print(f"DEBUG Viz: Plot de contagem de features GA, comprimento: {len(ga_features_history)}")

#         pso_features_history = self._get_history_data(self.pso_results, 'features_count')
#         if pso_features_history:
#             iterations_pso = list(range(len(pso_features_history)))
#             fig.add_trace(go.Scatter(x=iterations_pso, y=pso_features_history,
#                                      mode='lines+markers', name='Features Selecionadas (PSO)'))
#             print(f"DEBUG Viz: Plot de contagem de features PSO, comprimento: {len(pso_features_history)}")
        
#         if self.total_numerical_features_original > 0:
#             max_gens_iters = 0
#             if ga_features_history: max_gens_iters = max(max_gens_iters, len(ga_features_history))
#             if pso_features_history: max_gens_iters = max(max_gens_iters, len(pso_features_history))
            
#             if max_gens_iters > 0:
#                 fig.add_trace(go.Scatter(x=list(range(max_gens_iters)), 
#                                          y=[self.total_numerical_features_original] * max_gens_iters,
#                                          mode='lines', name='Total de Features Numéricas Originais', 
#                                          line=dict(dash='dot', color='grey')))

#         if not fig.data:
#             st.write("Nenhum dado de contagem de features selecionadas para plotar.")
#         else:
#             fig.update_layout(xaxis_title="Geração / Iteração", yaxis_title="Número de Features Numéricas Selecionadas")
#             st.plotly_chart(fig, use_container_width=True)
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

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
            self.plot_selected_features_count_over_time() # Chamada correta
        with tabs[2]:
            self.display_performance_metrics() # Chamada correta
        with tabs[3]:
            self.display_feature_selection_details() # Chamada correta
            
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
                yaxis_range=[0, 1.05] # Ajuste se o range do fitness for diferente
            )
            st.plotly_chart(fig, use_container_width=True)

    # ESTE MÉTODO E OS SEGUINTES DEVEM ESTAR INDENTADOS CORRETAMENTE
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
            if self.total_numerical_features_original > 0:
                reduction_ga = ((self.total_numerical_features_original - num_selected_ga) / self.total_numerical_features_original) * 100
                reduction_ga_text = f"{reduction_ga:.1f}% ({self.total_numerical_features_original - num_selected_ga} features reduzidas)"
            else:
                reduction_ga_text = "N/A (sem features numéricas originais)"
            
            ga_hist_len = len(self._get_history_data(self.ga_results, 'fitness'))
            data.append({
                "Algoritmo": "GA",
                "Melhor Fitness": f"{self.ga_results.get('best_fitness', float('inf')):.4f}",
                "Features Numéricas Selecionadas": num_selected_ga,
                "Total de Gerações": ga_hist_len -1 if ga_hist_len > 0 else "N/A"
            })
        
        if self.pso_results:
            num_selected_pso = self.pso_results.get('selected_features_count', 0)
            if self.total_numerical_features_original > 0:
                reduction_pso = ((self.total_numerical_features_original - num_selected_pso) / self.total_numerical_features_original) * 100
                reduction_pso_text = f"{reduction_pso:.1f}% ({self.total_numerical_features_original - num_selected_pso} features reduzidas)"
            else:
                reduction_pso_text = "N/A (sem features numéricas originais)"

            pso_hist_len = len(self._get_history_data(self.pso_results, 'fitness'))
            data.append({
                "Algoritmo": "PSO",
                "Melhor Fitness": f"{self.pso_results.get('best_fitness', float('inf')):.4f}",
                "Features Numéricas Selecionadas": num_selected_pso,
                "Total de Iterações": pso_hist_len -1 if pso_hist_len > 0 else "N/A"
            })
        
        if data:
            df_metrics = pd.DataFrame(data)
            st.table(df_metrics)

            cols = st.columns(2)
            with cols[0]:
                st.metric(label="Redução de Complexidade Numérica (GA)", value=reduction_ga_text)
            with cols[1]:
                st.metric(label="Redução de Complexidade Numérica (PSO)", value=reduction_pso_text)
        else:
            st.write("Nenhuma métrica de desempenho para exibir.")

    def display_feature_selection_details(self):
        st.subheader("Detalhes das Features Numéricas Selecionadas (Solução Final)")
        
        # GA Details
        if self.ga_results and 'best_solution_vector' in self.ga_results:
            st.markdown("#### Algoritmo Genético (GA)")
            ga_mask = np.array(self.ga_results['best_solution_vector'])
            st.markdown(f"**Máscara Binária:** `{ga_mask.astype(int)}`")
            
            ga_selected_indices = self.ga_results.get('selected_features_indices')
            if ga_selected_indices is not None: # Deve ser uma lista de índices
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


        # PSO Details
        if self.pso_results and 'best_solution_vector' in self.pso_results:
            st.markdown("#### Otimização por Enxame de Partículas (PSO)")
            pso_mask = np.array(self.pso_results['best_solution_vector'])
            st.markdown(f"**Máscara Binária:** `{pso_mask.astype(int)}`")

            pso_selected_indices = self.pso_results.get('selected_features_indices')
            if pso_selected_indices is not None: # Deve ser uma lista de índices
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