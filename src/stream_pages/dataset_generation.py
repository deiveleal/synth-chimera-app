"""
Dataset generation page module for Synth Chimera application.

This module contains the DatasetGenerationPage class that displays
generated datasets with both numeric and image data visualization.
"""

import streamlit as st
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, Union, Any
import torch


class DatasetGenerationPage:
    """
    Dataset generation page for displaying synthetic multimodal datasets.
    
    This class creates a comprehensive view of generated datasets including
    numeric features, image data, and class distributions with detailed
    statistics and sample visualizations.
    
    Args:
        X_num: Numeric features data (full dataset)
        X_img: Image data (full dataset) 
        y: Target labels (full dataset)
        n_samples: Original number of samples requested
        n_features: Original number of features requested
        X_num_train: Training set numeric features (optional)
        X_img_train: Training set image data (optional)
        y_train: Training set labels (optional)
        X_num_val: Validation set numeric features (optional)
        X_img_val: Validation set image data (optional)
        y_val: Validation set labels (optional)
    """
    
    def __init__(
        self, 
        X_num: Optional[Union[torch.Tensor, np.ndarray]], 
        X_img: Optional[Union[torch.Tensor, np.ndarray]], 
        y: Optional[Union[torch.Tensor, np.ndarray]], 
        n_samples: int, 
        n_features: int,
        X_num_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_img_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_num_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_img_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y_val: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        """Initialize and render the dataset generation page."""
        self.original_n_samples = n_samples
        self.original_n_features = n_features
        
        self._render_page(
            X_num, X_img, y, n_samples, n_features,
            X_num_train, X_img_train, y_train,
            X_num_val, X_img_val, y_val
        )
    
    def _render_page(
        self,
        X_num: Optional[Union[torch.Tensor, np.ndarray]], 
        X_img: Optional[Union[torch.Tensor, np.ndarray]], 
        y: Optional[Union[torch.Tensor, np.ndarray]], 
        n_samples: int, 
        n_features: int,
        X_num_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_img_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_num_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_img_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y_val: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> None:
        """Render the complete dataset generation page."""
        st.title("Dataset Generation")
        
        # Dataset selector for viewing different splits
        current_X_num, current_X_img, current_y, current_samples = self._handle_dataset_selection(
            X_num, X_img, y, n_samples,
            X_num_train, X_img_train, y_train,
            X_num_val, X_img_val, y_val
        )
        
        # Validate data availability
        if not self._validate_data(current_X_num, current_y):
            return
        
        # Convert data to numpy arrays
        X_num_np, y_np = self._convert_to_numpy(current_X_num, current_y)
        if X_num_np is None or y_np is None:
            return
        
        # Validate data dimensions
        if not self._validate_dimensions(X_num_np):
            return
        
        # Display dataset information and visualizations
        self._display_dataset_info(X_num_np, y_np, current_X_img, current_X_num, current_y)
    
    def _handle_dataset_selection(
        self,
        X_num: Optional[Union[torch.Tensor, np.ndarray]], 
        X_img: Optional[Union[torch.Tensor, np.ndarray]], 
        y: Optional[Union[torch.Tensor, np.ndarray]], 
        n_samples: int,
        X_num_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_img_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_num_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
        X_img_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y_val: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> tuple:
        """Handle dataset selection between full, training, and validation sets."""
        # Check if train/validation splits are available
        if X_num_train is not None and X_num_val is not None:
            dataset_view = st.selectbox(
                "Select dataset to view:",
                ["Complete Dataset", "Training Data", "Validation Data"]
            )
            
            if dataset_view == "Training Data":
                current_samples = len(y_train) if y_train is not None else 0
                st.info(f"Viewing training data: {current_samples} samples")
                return X_num_train, X_img_train, y_train, current_samples
            elif dataset_view == "Validation Data":
                current_samples = len(y_val) if y_val is not None else 0
                st.info(f"Viewing validation data: {current_samples} samples")
                return X_num_val, X_img_val, y_val, current_samples
            else:
                st.info(f"Viewing complete dataset: {n_samples} samples")
                return X_num, X_img, y, n_samples
        else:
            st.info(f"Viewing complete dataset: {n_samples} samples")
            return X_num, X_img, y, n_samples
    
    def _validate_data(
        self, 
        X_num: Optional[Union[torch.Tensor, np.ndarray]], 
        y: Optional[Union[torch.Tensor, np.ndarray]]
    ) -> bool:
        """Validate that required data is available."""
        if X_num is None or y is None:
            st.error("Data not available for visualization.")
            return False
        return True
    
    def _convert_to_numpy(
        self, 
        X_num: Union[torch.Tensor, np.ndarray], 
        y: Union[torch.Tensor, np.ndarray]
    ) -> tuple:
        """Convert data to numpy arrays with robust error handling."""
        try:
            X_num_np = X_num.cpu().numpy() if hasattr(X_num, 'cpu') else np.array(X_num)
            y_np = y.cpu().numpy() if hasattr(y, 'cpu') else np.array(y)
            return X_num_np, y_np
        except Exception as e:
            st.error(f"Error converting data: {e}")
            return None, None
    
    def _validate_dimensions(self, X_num_np: np.ndarray) -> bool:
        """Validate that numeric data has correct dimensions."""
        if len(X_num_np.shape) != 2:
            st.error("Numeric data must have 2 dimensions (samples x features)")
            return False
        return True
    
    def _display_dataset_info(
        self, 
        X_num_np: np.ndarray, 
        y_np: np.ndarray, 
        X_img: Optional[Union[torch.Tensor, np.ndarray]],
        original_X_num: Union[torch.Tensor, np.ndarray],
        original_y: Union[torch.Tensor, np.ndarray]
    ) -> None:
        """Display comprehensive dataset information and visualizations."""
        # Get actual dimensions from data
        actual_samples, actual_features = X_num_np.shape
        
        # Create feature names and dataframe
        feature_names = [f"Feature_{i+1}" for i in range(actual_features)]
        df_numeric = pd.DataFrame(X_num_np, columns=feature_names)
        df_numeric['Class'] = y_np
        
        # Display basic dataset information
        self._display_basic_info(actual_samples, actual_features, y_np)
        
        # Display numeric data sample
        self._display_numeric_sample(df_numeric)
        
        # Display image data sample
        self._display_image_section(X_img)
        
        # Display debug information
        self._display_debug_info(original_X_num, X_img, original_y)
    
    def _display_basic_info(self, actual_samples: int, actual_features: int, y_np: np.ndarray) -> None:
        """Display basic dataset information including class distribution."""
        st.subheader("Dataset Information")
        st.write(f"Number of samples: {actual_samples}")
        st.write(f"Number of numeric features: {actual_features}")
        
        # Display class distribution
        unique_classes, counts = np.unique(y_np, return_counts=True)
        st.write("Class distribution:")
        for class_val, count in zip(unique_classes, counts):
            percentage = count / len(y_np) * 100
            st.write(f"  - Class {class_val}: {count} samples ({percentage:.1f}%)")
    
    def _display_numeric_sample(self, df_numeric: pd.DataFrame) -> None:
        """Display sample of numeric data."""
        st.subheader("Numeric Data Sample")
        st.dataframe(df_numeric.head(10))
    
    def _display_image_section(self, X_img: Optional[Union[torch.Tensor, np.ndarray]]) -> None:
        """Display image data section."""
        st.subheader("Image Data Sample")
        if X_img is not None:
            self.display_image_samples(X_img)
        else:
            st.write("No image data available.")
    
    def _display_debug_info(
        self, 
        X_num: Union[torch.Tensor, np.ndarray], 
        X_img: Optional[Union[torch.Tensor, np.ndarray]], 
        y: Union[torch.Tensor, np.ndarray]
    ) -> None:
        """Display debug information in an expandable section."""
        with st.expander("🔧 Debug Information"):
            st.write("**Data types:**")
            st.write(f"- X_num: {type(X_num)}")
            st.write(f"- X_img: {type(X_img)}")
            st.write(f"- y: {type(y)}")
            
            st.write("**Shapes:**")
            st.write(f"- X_num shape: {X_num.shape if hasattr(X_num, 'shape') else 'N/A'}")
            st.write(f"- X_img shape: {X_img.shape if X_img is not None and hasattr(X_img, 'shape') else 'N/A'}")
            st.write(f"- y shape: {y.shape if hasattr(y, 'shape') else 'N/A'}")
    
    def display_image_samples(self, X_img: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Display sample images from the dataset.
        
        Args:
            X_img: Image data tensor or array
        """
        try:
            # Validate image data
            if X_img is None:
                st.write("Image data not available")
                return
            
            # Convert to numpy if necessary
            X_img_np = self._convert_images_to_numpy(X_img)
            if X_img_np is None:
                return
            
            if len(X_img_np) == 0:
                st.write("No images to display")
                return
            
            # Display sample images
            self._display_image_grid(X_img_np)
                    
        except Exception as e:
            st.error(f"Error displaying images: {e}")
            self._display_image_debug_info(X_img)
    
    def _convert_images_to_numpy(self, X_img: Union[torch.Tensor, np.ndarray]) -> Optional[np.ndarray]:
        """Convert image data to numpy array."""
        try:
            if hasattr(X_img, 'cpu'):
                return X_img.cpu().numpy()
            else:
                return np.array(X_img)
        except Exception as e:
            st.error(f"Error converting image data: {e}")
            return None
    
    def _display_image_grid(self, X_img_np: np.ndarray) -> None:
        """Display images in a grid layout."""
        num_samples = min(5, len(X_img_np))
        cols = st.columns(num_samples)
        
        for i in range(num_samples):
            with cols[i]:
                img = self._process_image_for_display(X_img_np[i])
                if img is not None:
                    st.image(img, caption=f"Image {i+1}", use_container_width=True)
    
    def _process_image_for_display(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Process individual image for display.
        
        Args:
            img: Single image array
            
        Returns:
            Processed image ready for display or None if processing fails
        """
        try:
            # Normalize values if necessary
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Adjust dimensions for display
            if len(img.shape) == 3:
                if img.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
                    img = img.transpose(1, 2, 0)
                
                if img.shape[-1] == 1:  # Remove single channel
                    img = img.squeeze(-1)
            
            return img
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    def _display_image_debug_info(self, X_img: Optional[Union[torch.Tensor, np.ndarray]]) -> None:
        """Display debug information for image data."""
        st.write("Debug details for troubleshooting:")
        if X_img is not None:
            try:
                shape_info = X_img.shape if hasattr(X_img, 'shape') else "Shape not available"
                st.write(f"Image data shape: {shape_info}")
            except:
                st.write("Could not retrieve image information")
    
    def get_dataset_summary(self) -> dict:
        """
        Get summary information about the dataset.
        
        Returns:
            Dictionary with dataset summary information
        """
        return {
            "requested_samples": self.original_n_samples,
            "requested_features": self.original_n_features,
            "page_type": "Dataset Generation",
            "supports_train_val_split": True,
            "supports_image_data": True
        }


def main():
    """Main function for running the Dataset Generation page standalone."""
    st.write("Dataset Generation Page - requires dataset parameters")


if __name__ == "__main__":
    main()