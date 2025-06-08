import sys
from utils.memory_utils import clear_gpu_memory

# Limpar memória GPU antes de importar outros módulos pesados
print("Liberando memória GPU antes de iniciar...")
clear_gpu_memory()

import streamlit as st
from stream_pages.about import AboutPage
from stream_pages.dataset_generation import DatasetGenerationPage
from stream_pages.feature_selection import FeatureSelectionPage
from stream_pages.results_visualization import ResultsVisualizationPage

# from components import sidebar, visualizations
from utils.device_detection import get_available_device
from utils.generate_dataset import generate_multimodal_dataset


def main():
    st.set_page_config(page_title="Synth Chimera",
                       page_icon=":brain:", layout="wide")
    st.header("Synth Chimera")
    tab_home, tab_dataset, tab_results = st.tabs([
        "Home",
        "Dataset Generation",
        "Results Visualization"
    ])

    # # ---- HIDE STREAMLIT STYLE ----
    # hide_st_style = """
    #             <style>
    #             #MainMenu {visibility: hidden;}
    #             footer {visibility: hidden;}
    #             header {visibility: hidden;}
    #             </style>
    #             """
    # st.markdown(hide_st_style, unsafe_allow_html=True)

    # Variables
    X_num = None
    X_img = None
    y = None
    n_samples = None
    n_features = None
    n_classes = None
    num_pop = None
    num_gen = None
    num_part = None
    num_iter = None

    # Inicializar estado da sessão para armazenar resultados
    if 'ga_results' not in st.session_state:
        st.session_state.ga_results = None
    if 'pso_results' not in st.session_state:
        st.session_state.pso_results = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None

    # ---- SIDEBAR ----

    st.sidebar.header("Settings")
    device = get_available_device()
    st.sidebar.write(f"Device in usage: {device}")

    # User inputs for dataset generation
    st.sidebar.subheader("Dataset Generation Parameters")
    n_samples = st.sidebar.number_input(
        "Number of Records:", min_value=1, value=2500)
    n_features = st.sidebar.number_input(
        "Number of Features (Even Number - Except Image):", min_value=2, value=10)
    n_classes = st.sidebar.number_input(
        "Number of Classes(Number of records must be divisible):", min_value=1, value=5)

    st.sidebar.subheader("Feature Selection Parameters")
    # User inputs for feature selection
    num_pop = st.sidebar.number_input(
        "GA: Initial Population:", min_value=1, value=30)
    num_gen = st.sidebar.number_input(
        "GA: Number of Generations:", min_value=1, value=5)
    num_part = st.sidebar.number_input(
        "PSO: Number of Particles:", min_value=1, value=30)
    num_iter = st.sidebar.number_input(
        "PSO: Number of Iterations:", min_value=1, value=5)
    
    # Função de callback para salvar resultados
    def save_results(algorithm_type, results):
        if algorithm_type == "GA":
            st.session_state.ga_results = results
        elif algorithm_type == "PSO":
            st.session_state.pso_results = results

    with tab_home:
        if st.sidebar.button("Run Feature Selection"):
            if n_samples > 0 and n_features > 0 and n_classes > 0:
                X_num, X_img, y = generate_multimodal_dataset(
                    num_samples=n_samples,
                    num_features=n_features,
                    num_classes=n_classes
                )
                X_num, X_img, y = X_num.to(device), X_img.to(device), y.to(device)
                
                # Gerar nomes de features
                feature_names = [f"Feature_{i+1}" for i in range(n_features)]
                feature_names.append("Image_Feature")
                st.session_state.feature_names = feature_names

                st.title("Feature Selection Running")
                FeatureSelectionPage(
                    num_pop=num_pop,
                    num_gen=num_gen,
                    num_part=num_part,
                    num_iter=num_iter,
                    X_num=X_num,
                    X_img=X_img,
                    y=y,
                    save_results_callback=save_results
                )
            else:
                st.error("Please fill in all parameters correctly.")
        else:
            AboutPage()

    with tab_dataset:
        if X_num is not None and X_img is not None and y is not None:
            DatasetGenerationPage(
                X_num=X_num,
                X_img=X_img,
                y=y,
                n_samples=n_samples,
                n_features=n_features
            )
        else:
            st.warning("Dataset not created yet. Please generate a dataset first.")

    with tab_results:
        if (st.session_state.ga_results is not None or 
            st.session_state.pso_results is not None):
            ResultsVisualizationPage(
                ga_results=st.session_state.ga_results,
                pso_results=st.session_state.pso_results,
                feature_names=st.session_state.feature_names
            )
        else:
            st.warning("No results to display. Run the algorithm first.")


if __name__ == "__main__":
    main()
