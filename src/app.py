# -*- coding: utf-8 -*-
import os
from utils.memory_utils import clear_gpu_memory


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("Liberando memória GPU antes de iniciar...")
clear_gpu_memory()

import streamlit as st
from stream_pages.about import AboutPage
from stream_pages.dataset_generation import DatasetGenerationPage
from stream_pages.feature_selection import FeatureSelectionPage
from stream_pages.results_visualization import ResultsVisualizationPage

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
    if 'ga_results' not in st.session_state:
        st.session_state.ga_results = None
    if 'pso_results' not in st.session_state:
        st.session_state.pso_results = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'dataset_generated' not in st.session_state:
        st.session_state.dataset_generated = False
        st.session_state.X_num = None
        st.session_state.X_img = None
        st.session_state.y = None
        st.session_state.n_samples_generated = None
        st.session_state.n_features_generated = None
        st.session_state.n_classes_generated = None


    # ---- SIDEBAR ----
    st.sidebar.header("Settings")
    device = get_available_device()
    st.sidebar.write(f"Device in usage: {device}")

    # User inputs for dataset generation
    st.sidebar.subheader("Dataset Generation Parameters")
    n_samples_input = st.sidebar.number_input(
        "Number of Records:", min_value=1, value=st.session_state.get('n_samples_generated', 3000))
    n_features_input = st.sidebar.number_input(
        "Number of Features (Even Number - Except Image):", min_value=2, value=st.session_state.get('n_features_generated', 10))
    n_classes_input = st.sidebar.slider(
    "Number of Classes (max 7):", min_value=2, max_value=7, value=st.session_state.get('n_classes_generated', 5))

    if st.sidebar.button("Generate Dataset"):
        if n_samples_input > 0 and n_features_input > 0 and n_classes_input > 0:
            with st.spinner("Generating dataset..."):
                st.session_state.X_num, st.session_state.X_img, st.session_state.y = generate_multimodal_dataset(
                    num_samples=n_samples_input,
                    num_features=n_features_input,
                    num_classes=n_classes_input
                )
            st.session_state.n_samples_generated = n_samples_input
            st.session_state.n_features_generated = n_features_input
            st.session_state.n_classes_generated = n_classes_input
            st.session_state.dataset_generated = True
            feature_names = [f"Feature_{i+1}" for i in range(st.session_state.n_features_generated)]
            feature_names.append("Image_Feature") 
            st.session_state.feature_names = feature_names
            st.sidebar.success("Dataset generated successfully!")
        else:
            st.sidebar.error("Please set valid dataset parameters.")


    st.sidebar.subheader("Feature Selection Parameters")
    num_pop = st.sidebar.number_input(
        "GA: Initial Population:", min_value=1, value=30)
    num_gen = st.sidebar.number_input(
        "GA: Number of Generations:", min_value=1, value=5)
    num_part = st.sidebar.number_input(
        "PSO: Number of Particles:", min_value=1, value=30)
    num_iter = st.sidebar.number_input(
        "PSO: Number of Iterations:", min_value=1, value=5)
    
    def save_results(algorithm_type, results):
        if algorithm_type == "GA":
            st.session_state.ga_results = results
        elif algorithm_type == "PSO":
            st.session_state.pso_results = results

    with tab_home:
        if st.sidebar.button("Run Feature Selection"):
            if st.session_state.dataset_generated:
                X_num_device = st.session_state.X_num.to(device)
                X_img_device = st.session_state.X_img.to(device)
                y_device = st.session_state.y.to(device)
                
                st.title("Feature Selection Running")
                with st.spinner("Running feature selection algorithms... This may take a while."):
                    FeatureSelectionPage(
                        num_pop=num_pop,
                        num_gen=num_gen,
                        num_part=num_part,
                        num_iter=num_iter,
                        X_num=X_num_device,
                        X_img=X_img_device,
                        y=y_device,
                        save_results_callback=save_results
                    )
                st.success("Feature selection completed!")
                st.rerun()
            else:
                st.error("Please generate a dataset first using the 'Generate Dataset' button in the sidebar.")
        else:
            AboutPage()

    with tab_dataset:
        if st.session_state.dataset_generated:
            DatasetGenerationPage(
                X_num=st.session_state.X_num,
                X_img=st.session_state.X_img,
                y=st.session_state.y,
                n_samples=st.session_state.n_samples_generated,
                n_features=st.session_state.n_features_generated
            )
        else:
            st.warning("Dataset not created yet. Please use the 'Generate Dataset' button in the sidebar.")

    with tab_results:
        if (st.session_state.ga_results is not None or 
            st.session_state.pso_results is not None):
            ResultsVisualizationPage(
                ga_results=st.session_state.ga_results,
                pso_results=st.session_state.pso_results,
                numerical_feature_names=st.session_state.feature_names[:-1],
            )
        else:
            st.warning("No results to display. Generate a dataset and run the feature selection first.")


if __name__ == "__main__":
    main()