# -*- coding: utf-8 -*-
"""
Main application module for Synth Chimera.

This module contains the main Streamlit application that orchestrates
dataset generation, feature selection, and results visualization.
"""

import os
from utils.memory_utils import clear_gpu_memory
from sklearn.model_selection import train_test_split  # type: ignore
import torch

# Configure PyTorch CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("Clearing GPU memory before starting...")
clear_gpu_memory()

import streamlit as st
from stream_pages.about import AboutPage
from stream_pages.dataset_generation import DatasetGenerationPage
from stream_pages.feature_selection import FeatureSelectionPage
from stream_pages.results_visualization import ResultsVisualizationPage

from utils.device_detection import get_available_device
from utils.generate_dataset import generate_multimodal_dataset


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    session_vars = {
        'ga_results': None,
        'pso_results': None,
        'feature_names': None,
        'dataset_generated': False,
        'X_num': None,
        'X_img': None,
        'y': None,
        'n_samples_generated': None,
        'n_features_generated': None,
        'n_classes_generated': None,
        'X_num_train': None,
        'X_num_val': None,
        'X_img_train': None,
        'X_img_val': None,
        'y_train': None,
        'y_val': None
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


def create_sidebar_controls() -> tuple:
    """
    Create sidebar controls for dataset generation and feature selection.
    
    Returns:
        Tuple containing dataset parameters and feature selection parameters
    """
    st.sidebar.header("Settings")
    device = get_available_device()
    st.sidebar.write(f"Device: {device}")

    # Dataset generation parameters
    st.sidebar.subheader("Dataset Generation Parameters")
    n_samples_input = st.sidebar.number_input(
        "Number of Samples:", min_value=1, value=2500)
    n_features_input = st.sidebar.number_input(
        "Number of Features (Even Number - Except Image):", 
        min_value=2, value=10, step=2)
    n_classes_input = st.sidebar.number_input(
        "Number of Classes (max 7):", 
        min_value=2, max_value=7, value=2, step=1)
    
    # Data split parameters
    st.sidebar.subheader("Data Split Parameters")
    val_percent = st.sidebar.number_input(
        "Validation Data Percentage",
        min_value=10,
        max_value=90,
        value=70,
        step=5,
        help="Percentage of data reserved for validation (remaining will be used for training)"
    )
    train_percent = 100 - val_percent
    st.sidebar.write(f"Training: {train_percent}% | Validation: {val_percent}%")

    # Dataset generation button - positioned here before Feature Selection Parameters
    if st.sidebar.button("Generate Dataset", type="primary"):
        generate_dataset(n_samples_input, n_features_input, n_classes_input, val_percent)

    # Feature selection parameters
    st.sidebar.subheader("Feature Selection Parameters")
    num_pop = st.sidebar.number_input(
        "GA: Initial Population:", min_value=1, value=30)
    num_gen = st.sidebar.number_input(
        "GA: Number of Generations:", min_value=1, value=5)
    num_part = st.sidebar.number_input(
        "PSO: Number of Particles:", min_value=1, value=30)
    num_iter = st.sidebar.number_input(
        "PSO: Number of Iterations:", min_value=1, value=5)
    
    return (n_samples_input, n_features_input, n_classes_input, val_percent,
            num_pop, num_gen, num_part, num_iter, device)


def display_dataset_info() -> None:
    """Display dataset information in the sidebar."""
    if st.session_state.dataset_generated:
        st.sidebar.write("---")
        st.sidebar.subheader("Dataset Info")
        if hasattr(st.session_state, 'y_train') and st.session_state.y_train is not None:
            st.sidebar.write(f"🎯 Training: {len(st.session_state.y_train)} samples")
            st.sidebar.write(f"🎯 Validation: {len(st.session_state.y_val)} samples")
        st.sidebar.write(f"📊 Features: {st.session_state.n_features_generated}")
        st.sidebar.write(f"🏷️ Classes: {st.session_state.n_classes_generated}")


def generate_dataset(n_samples: int, n_features: int, n_classes: int, val_percent: int) -> None:
    """
    Generate and split the multimodal dataset.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        n_classes: Number of classes to generate
        val_percent: Percentage of data for validation
    """
    if n_samples > 0 and n_features > 0 and n_classes > 0:
        with st.spinner("Generating dataset..."):
            # Generate dataset
            st.session_state.X_num, st.session_state.X_img, st.session_state.y = generate_multimodal_dataset(
                num_samples=n_samples,
                num_features=n_features,
                num_classes=n_classes
            )
            
            # Convert to numpy for splitting
            X_num = (st.session_state.X_num.cpu().numpy() 
                    if hasattr(st.session_state.X_num, "cpu") 
                    else st.session_state.X_num)
            X_img = (st.session_state.X_img.cpu().numpy() 
                    if hasattr(st.session_state.X_img, "cpu") 
                    else st.session_state.X_img)
            y = (st.session_state.y.cpu().numpy() 
                if hasattr(st.session_state.y, "cpu") 
                else st.session_state.y)
            
            # Perform train/validation split
            test_size = val_percent / 100.0
            X_num_train, X_num_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
                X_num, X_img, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Convert back to tensors
            st.session_state.X_num_train = torch.tensor(X_num_train, dtype=torch.float32)
            st.session_state.X_num_val = torch.tensor(X_num_val, dtype=torch.float32)
            st.session_state.X_img_train = torch.tensor(X_img_train, dtype=torch.float32)
            st.session_state.X_img_val = torch.tensor(X_img_val, dtype=torch.float32)
            st.session_state.y_train = torch.tensor(y_train, dtype=torch.long)
            st.session_state.y_val = torch.tensor(y_val, dtype=torch.long)
            
            # Keep full dataset for compatibility
            st.session_state.X_num = torch.tensor(X_num, dtype=torch.float32)
            st.session_state.X_img = torch.tensor(X_img, dtype=torch.float32)
            st.session_state.y = torch.tensor(y, dtype=torch.long)
            
        # Update session state
        st.session_state.n_samples_generated = n_samples
        st.session_state.n_features_generated = n_features
        st.session_state.n_classes_generated = n_classes
        st.session_state.dataset_generated = True
        
        # Create feature names
        feature_names = [f"Feature_{i+1}" for i in range(st.session_state.n_features_generated)]
        feature_names.append("Image_Feature") 
        st.session_state.feature_names = feature_names
        
        st.sidebar.success(
            f"Dataset generated successfully! "
            f"Training: {len(y_train)} samples, Validation: {len(y_val)} samples"
        )
    else:
        st.sidebar.error("Please set valid dataset parameters.")


def save_results(algorithm_type: str, results: dict) -> None:
    """
    Save feature selection results to session state.
    
    Args:
        algorithm_type: Type of algorithm ('GA' or 'PSO')
        results: Results dictionary from the algorithm
    """
    if algorithm_type == "GA":
        st.session_state.ga_results = results
    elif algorithm_type == "PSO":
        st.session_state.pso_results = results


def run_feature_selection(device: str, num_pop: int, num_gen: int, 
                         num_part: int, num_iter: int) -> None:
    """
    Run feature selection algorithms on the generated dataset.
    
    Args:
        device: Device to run computations on
        num_pop: GA population size
        num_gen: GA number of generations
        num_part: PSO number of particles
        num_iter: PSO number of iterations
    """
    if st.session_state.dataset_generated:
        # Prepare training data
        X_num_device = st.session_state.X_num_train.to(device)
        X_img_device = st.session_state.X_img_train.to(device)
        y_device = st.session_state.y_train.to(device)
        
        # Prepare validation data
        X_num_val_device = st.session_state.X_num_val.to(device)
        X_img_val_device = st.session_state.X_img_val.to(device)
        y_val_device = st.session_state.y_val.to(device)

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
                X_num_val=X_num_val_device,
                X_img_val=X_img_val_device,
                y_val=y_val_device,
                save_results_callback=save_results
            )
        st.success("Feature selection completed!")
        st.rerun()
    else:
        st.error("Please generate a dataset first using the 'Generate Dataset' button in the sidebar.")


def main():
    """Main application function."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Synth Chimera",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Create main header and tabs
    st.header("Synth Chimera")
    tab_home, tab_dataset, tab_results = st.tabs([
        "Home",
        "Dataset Generation",
        "Results Visualization"
    ])
    
    # Create sidebar controls - now includes the Generate Dataset button
    (n_samples_input, n_features_input, n_classes_input, val_percent,
     num_pop, num_gen, num_part, num_iter, device) = create_sidebar_controls()
    
    # Display dataset info
    # display_dataset_info()
    
    # Home tab - Feature selection or About page
    with tab_home:
        if st.sidebar.button("Run Feature Selection", type="primary"):
            run_feature_selection(device, num_pop, num_gen, num_part, num_iter)
        else:
            AboutPage()

    # Dataset Generation tab
    with tab_dataset:
        if st.session_state.dataset_generated:
            DatasetGenerationPage(
                X_num=st.session_state.X_num,
                X_img=st.session_state.X_img,
                y=st.session_state.y,
                n_samples=st.session_state.n_samples_generated,
                n_features=st.session_state.n_features_generated,
                # Add training and validation sets for visualization
                X_num_train=st.session_state.X_num_train,
                X_img_train=st.session_state.X_img_train,
                y_train=st.session_state.y_train,
                X_num_val=st.session_state.X_num_val,
                X_img_val=st.session_state.X_img_val,
                y_val=st.session_state.y_val
            )
        else:
            st.warning("Dataset not created yet. Please use the 'Generate Dataset' button in the sidebar.")

    # Results Visualization tab
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