# Synth Chimera App

## Descrição
O Synth Chimera App é uma aplicação desenvolvida com Streamlit que permite a geração de datasets multimodais, seleção de características e visualização de resultados. O aplicativo oferece uma interface amigável para usuários interagirem com diferentes métodos de otimização e avaliação de características.

## Estrutura do Projeto
O projeto é organizado da seguinte forma:

```
synth-chimera-app
├── src
│   ├── app.py                     # Ponto de entrada da aplicação Streamlit
│   ├── pages                      # Contém as diferentes páginas do aplicativo
│   │   ├── __init__.py            # Inicializa o pacote de páginas
│   │   ├── dataset_generation.py   # Interface para geração de datasets
│   │   ├── feature_selection.py    # Interface para seleção de características
│   │   ├── results_visualization.py # Visualização dos resultados da avaliação
│   │   └── about.py               # Informações sobre o aplicativo
│   ├── components                  # Contém componentes reutilizáveis
│   │   ├── __init__.py            # Inicializa o pacote de componentes
│   │   ├── sidebar.py             # Define a barra lateral de navegação
│   │   └── visualizations.py       # Funções para criar visualizações
│   ├── utils                       # Contém funções utilitárias
│   │   ├── __init__.py            # Inicializa o pacote de utilitários
│   │   ├── generate_dataset.py     # Geração de datasets multimodais
│   │   ├── cnn_fitness.py          # Avaliação das características
│   │   ├── optimization.py          # Algoritmos de otimização
│   │   ├── device_detection.py      # Detecção de dispositivos disponíveis
│   │   └── save_dataset.py         # Função para salvar resultados em Excel
│   └── models                      # Contém modelos (se necessário)
│       └── __init__.py            # Inicializa o pacote de modelos
├── requirements.txt                # Dependências do projeto
├── .gitignore                      # Arquivos a serem ignorados pelo controle de versão
└── README.md                       # Documentação do projeto
```

## Instalação
Para instalar as dependências do projeto, execute o seguinte comando:

```
pip install -r requirements.txt
```

## Uso
Para iniciar a aplicação, execute o seguinte comando:

```
streamlit run src/app.py
```

Acesse a aplicação no seu navegador através do endereço indicado no terminal.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## Licença
Este projeto está licenciado sob a MIT License. Veja o arquivo LICENSE para mais detalhes.