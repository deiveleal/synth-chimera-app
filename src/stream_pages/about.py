import streamlit as st


class AboutPage:
    def __init__(self):
        st.title("Sobre o Synth Chimera")
        st.write(
            """
          O Synth Chimera é uma aplicação projetada para facilitar a geração de
               datasets multimodais e a seleção de características utilizando
               algoritmos de otimização.
          
          ## Funcionalidades
          - **Geração de Datasets**: Permite que os usuários especifiquem
               parâmetros como número de amostras, características e classes
               para criar datasets personalizados.
          - **Seleção de Características**: Oferece métodos como Algoritmo
               Genético (GA) e Otimização por Enxame de Partículas (PSO) para
               selecionar as características mais relevantes.
          - **Visualização de Resultados**: Apresenta gráficos e tabelas que
               mostram as métricas de desempenho dos modelos avaliados.

          ## Como Usar
          1. Navegue até a página de geração de datasets para criar um novo
               dataset.
          2. Utilize a página de seleção de características para aplicar os
               métodos de otimização.
          3. Visualize os resultados na página de visualização.

          O Synth Chimera é uma ferramenta poderosa para pesquisadores e
               desenvolvedores que desejam explorar e otimizar modelos de 
               aprendizado de máquina.
            """)


if __name__ == "__main__":
    AboutPage()
