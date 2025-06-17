# Synth-Chimera-App: Seleção de Features Multimodais com Algoritmos Bio-inspirados e CNNs

## Visão Geral

O **Synth-Chimera-App** é uma aplicação desenvolvida em Python com Streamlit que permite a seleção de características (features) em conjuntos de dados multimodais (numéricos e de imagem). O projeto utiliza Algoritmos Genéticos (GA) e Otimização por Enxame de Partículas (PSO) para identificar os subconjuntos de features mais relevantes. A avaliação da "qualidade" (fitness) de cada subconjunto é realizada através do treinamento e avaliação de uma Rede Neural Convolucional (CNN) multimodal.

A interface do Streamlit facilita o upload de dados, a configuração dos parâmetros dos algoritmos de otimização, a execução do processo de seleção e a visualização dos resultados, incluindo métricas de desempenho e gráficos da evolução da otimização.

## Funcionalidades Principais

*   **Interface Interativa:** Construída com Streamlit para fácil utilização.
*   **Suporte a Dados Multimodais:** Permite o upload de dados numéricos (CSV) e dados de imagem.
*   **Algoritmos de Seleção de Features:**
    *   Algoritmo Genético (GA)
    *   Otimização por Enxame de Partículas (PSO)
*   **Avaliação de Fitness Baseada em CNN:** Uma CNN multimodal customizada é treinada para avaliar o desempenho de cada subconjunto de features.
*   **Cache de Avaliação:** Mecanismo de cache para armazenar resultados de fitness já calculados, acelerando execuções repetidas ou avaliações de subconjuntos idênticos.
*   **Configuração Flexível:** Permite ajustar diversos parâmetros dos algoritmos GA e PSO, bem como as épocas de treinamento da CNN de avaliação.
*   **Visualização de Resultados:**
    *   Métricas de desempenho final (e.g., acurácia com features selecionadas).
    *   Contagem de features selecionadas e percentual de redução.
    *   Gráficos da evolução do fitness (melhor e média) ao longo das gerações/iterações.
    *   Gráficos da contagem de features selecionadas ao longo do processo.
    *   Exibição da máscara binária das features selecionadas.

## Estrutura do Projeto (Principais Componentes)

```
synth-chimera-app/
├── src/
│   ├── models/
│   │   └── cnn_model.py            # Definição da arquitetura da MultimodalCNN
│   ├── stream_pages/
│   │   ├── feature_selection.py    # Página Streamlit para configuração e execução da seleção
│   │   └── results_visualization.py # Página Streamlit para visualização dos resultados
│   ├── utils/
│   │   ├── cnn_fitness.py          # Lógica da função de fitness (evaluate_features) e cache
│   │   ├── data_processing.py      # (Suposto) Utilitários para pré-processamento de dados
│   │   └── optimization.py         # Implementações do Algoritmo Genético e PSO
│   └── app.py                      # Script principal da aplicação Streamlit
├── data/                           # (Opcional) Diretório para exemplos de dados
│   └── ...
├── .venv/                          # Diretório do ambiente virtual (se utilizado)
├── requirements.txt                # Lista de dependências do projeto
└── README.md                       # Este arquivo
```

## Como Funciona

1.  **Upload de Dados:** O usuário fornece dados numéricos (CSV) e, opcionalmente, um diretório com imagens e um arquivo CSV mapeando imagens para amostras.
2.  **Configuração:** O usuário define qual algoritmo de seleção de features utilizar (GA ou PSO) e configura seus respectivos hiperparâmetros (tamanho da população, número de gerações/iterações, taxas de mutação/crossover, parâmetros do PSO, etc.), incluindo o número de épocas para o treinamento da CNN na função de fitness.
3.  **Processo de Otimização:**
    *   O algoritmo escolhido (GA ou PSO) gera subconjuntos de features candidatas (indivíduos/partículas).
    *   Para cada subconjunto, a função `evaluate_features` é chamada.
    *   `evaluate_features` treina uma `MultimodalCNN` utilizando apenas as features do subconjunto atual (e dados de imagem, se aplicável).
    *   A acurácia (ou outra métrica) do modelo treinado é retornada como o valor de fitness do subconjunto.
    *   Resultados de fitness são cacheados para evitar recálculos.
    *   O algoritmo bio-inspirado utiliza os valores de fitness para guiar a busca por melhores subconjuntos ao longo das gerações/iterações.
4.  **Resultados:** Ao final do processo, a aplicação exibe:
    *   O melhor subconjunto de features encontrado.
    *   A acurácia alcançada com este subconjunto.
    *   Métricas de redução de dimensionalidade.
    *   Gráficos detalhando a progressão da otimização.

## Requisitos

*   Python (testado com 3.10+, recomendado 3.12)
*   Bibliotecas listadas em `requirements.txt`. Principais incluem:
    *   `streamlit`
    *   `torch` (PyTorch)
    *   `torchvision`
    *   `numpy`
    *   `pandas`
    *   `scikit-learn`
    *   `plotly` (para gráficos)

## Instalação e Configuração

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/synth-chimera-app.git
    cd synth-chimera-app
    ```

2.  **Crie e ative um ambiente virtual** (recomendado):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Linux/macOS
    # .venv\Scripts\activate    # No Windows
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota sobre PyTorch:* Se você precisar de uma versão específica do PyTorch (e.g., com suporte a CUDA), pode ser necessário instalá-lo separadamente seguindo as instruções em [pytorch.org](https://pytorch.org/).

## Como Executar

1.  Certifique-se de que seu ambiente virtual está ativado.
2.  Navegue até o diretório `src` do projeto:
    ```bash
    cd src
    ```
3.  Execute a aplicação Streamlit:
    ```bash
    streamlit run app.py
    ```
4.  Abra o navegador e acesse o endereço local fornecido pelo Streamlit (geralmente `http://localhost:8501`).

## Principais Módulos e Suas Responsabilidades

*   **`src/app.py`**: Ponto de entrada da aplicação Streamlit. Gerencia a navegação entre as diferentes páginas (Configuração, Visualização de Resultados).
*   **`src/models/cnn_model.py`**: Define a arquitetura da `MultimodalCNN`, capaz de processar dados numéricos e de imagem.
*   **`src/utils/cnn_fitness.py`**:
    *   `evaluate_features()`: Função crucial que atua como a função de fitness. Recebe um subconjunto de features, treina a `MultimodalCNN` com essas features e retorna uma métrica de desempenho (e.g., acurácia).
    *   Gerencia o `EVALUATION_CACHE` para armazenar e recuperar resultados de avaliações anteriores.
*   **`src/utils/optimization.py`**: Contém as implementações do `genetic_algorithm()` e `particle_swarm_optimization()`, incluindo suas lógicas de inicialização, avaliação, seleção, operadores de variação e atualização.
*   **`src/stream_pages/feature_selection.py`**: Constrói a interface Streamlit onde os usuários podem carregar dados, configurar os parâmetros dos algoritmos de otimização (GA/PSO) e iniciar o processo de seleção de features.
*   **`src/stream_pages/results_visualization.py`**: Responsável por apresentar os resultados da seleção de features de forma organizada, incluindo tabelas de métricas, máscaras de features selecionadas e gráficos interativos da evolução do processo de otimização.

## Possíveis Melhorias Futuras

*   Suporte a diferentes métricas de fitness além da acurácia.
*   Implementação de mais algoritmos de seleção de features.
*   Opções avançadas de pré-processamento de dados na interface.
*   Paralelização da avaliação de fitness para acelerar o processo.
*   Testes unitários e de integração mais abrangentes.
*   Empacotamento da aplicação (e.g., com Docker).

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.


---
*Este README foi atualizado em 16 de junho de 2025.*
```