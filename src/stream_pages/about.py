"""
About page module for Synth Chimera application.

This module contains the AboutPage class that displays information about
the Synth Chimera application, its features, and usage instructions.
"""

import streamlit as st


class AboutPage:
    """
    About page class for displaying application information.
    
    This class creates a Streamlit page that provides comprehensive information
    about the Synth Chimera application, including its features, functionality,
    and usage instructions.
    """
    
    def __init__(self):
        """Initialize and render the About page."""
        self._render_page()
    
    def _render_page(self) -> None:
        """Render the complete About page content."""
        self._render_header()
        self._render_overview()
        self._render_features()
        self._render_usage_guide()
        self._render_footer()
    
    def _render_header(self) -> None:
        """Render the page header."""
        st.title("About Synth Chimera")
    
    def _render_overview(self) -> None:
        """Render the application overview section."""
        st.markdown(
            """
            Synth Chimera is an advanced application designed to facilitate the generation of
            multimodal datasets and feature selection using state-of-the-art optimization
            algorithms. 
            
            This powerful tool enables researchers and developers to create
            synthetic datasets and optimize machine learning models efficiently.
            """
        )
    
    def _render_features(self) -> None:
        """Render the features section."""
        st.markdown("## Features")
        
        feature_descriptions = {
            "Dataset Generation": (
                "Allows users to specify parameters such as number of samples, "
                "features, and classes to create customized multimodal datasets "
                "with both structured and image data."
            ),
            "Feature Selection": (
                "Offers advanced optimization methods including Genetic Algorithm (GA) "
                "and Particle Swarm Optimization (PSO) to select the most relevant "
                "features for improved model performance."
            ),
            "Results Visualization": (
                "Provides comprehensive charts and tables displaying performance "
                "metrics of evaluated models, enabling easy analysis and comparison "
                "of different configurations."
            ),
            "Multimodal Support": (
                "Handles both structured data and image data simultaneously, "
                "allowing for complex multimodal machine learning experiments."
            ),
            "Interactive Interface": (
                "Features an intuitive Streamlit-based web interface that makes "
                "complex machine learning tasks accessible to users of all levels."
            )
        }
        
        for feature, description in feature_descriptions.items():
            st.markdown(f"- **{feature}**: {description}")
    
    def _render_usage_guide(self) -> None:
        """Render the usage guide section."""
        st.markdown("## How to Use")
        
        usage_steps = [
            "Navigate to the **Dataset Generation** page to create a new synthetic dataset with your desired parameters.",
            "Use the **Feature Selection** page to apply optimization methods (GA or PSO) for identifying the most relevant features.",
            "View and analyze results on the **Results Visualization** page to understand model performance and feature importance.",
            "Experiment with different configurations to find the optimal setup for your specific use case.",
            "Export or save your results for further analysis or integration into your machine learning pipeline."
        ]
        
        for i, step in enumerate(usage_steps, 1):
            st.markdown(f"{i}. {step}")
    
    def _render_footer(self) -> None:
        """Render the footer section."""
        st.markdown("---")
        st.markdown(
            """
            **Synth Chimera** is a powerful tool for researchers and developers who want to
            explore and optimize machine learning models through synthetic data generation
            and intelligent feature selection.
            
            Whether you're conducting academic research
            or developing production systems, Synth Chimera provides the tools you need
            to enhance your machine learning workflows.
            
            *Built with Python, Streamlit, PyTorch, and advanced optimization algorithms.*
            """
        )
        
        # Optional: Add technical details in an expander
        with st.expander("Technical Details"):
            st.markdown(
                """
                ### Technologies Used
                - **Frontend**: Streamlit for interactive web interface
                - **Backend**: Python with PyTorch for deep learning models
                - **Optimization**: Genetic Algorithm (GA) and Particle Swarm Optimization (PSO)
                - **Data Processing**: NumPy, Pandas for data manipulation
                - **Visualization**: Matplotlib, Plotly for charts and graphs
                
                ### Supported Data Types
                - Structured/Tabular data
                - Image data (various formats)
                - Multimodal combinations
                
                ### Model Types
                - Convolutional Neural Networks (CNN)
                - Multimodal Neural Networks
                - Custom architectures for specific use cases
                """
            )


def main():
    """Main function for running the About page standalone."""
    AboutPage()


if __name__ == "__main__":
    main()