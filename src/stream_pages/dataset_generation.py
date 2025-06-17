import streamlit as st
import pandas as pd # type: ignore
import numpy as np

class DatasetGenerationPage:
    def __init__(self, X_num, X_img, y, n_samples, n_features):
        st.title("Dataset Generation")
        
        X_num_np = X_num.cpu().numpy() if hasattr(X_num, 'cpu') else np.array(X_num)
        y_np = y.cpu().numpy() if hasattr(y, 'cpu') else np.array(y)
        
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        df_numeric = pd.DataFrame(X_num_np, columns=feature_names)
        df_numeric['Class'] = y_np
        
        st.subheader("Dataset Information")
        st.write(f"Number of samples: {n_samples}")
        st.write(f"Number of numeric features: {n_features}")
        
        st.subheader("Numeric Data Sample")
        st.dataframe(df_numeric.head(10))
        
        st.subheader("Image Data Sample")
        if hasattr(X_img, 'cpu'):
            self.display_image_samples(X_img)
    
    def display_image_samples(self, X_img):
        try:
            num_samples = min(5, len(X_img))
            cols = st.columns(num_samples)
            
            for i in range(num_samples):
                with cols[i]:
                    img = X_img[i].cpu().numpy()
                    
                    if len(img.shape) > 2:
                        if img.shape[0] in (1, 3):
                            img = img.transpose(1, 2, 0)
                        
                        if img.shape[-1] == 1:
                            img = img.squeeze()
                    
                    st.image(img, caption=f"Image {i+1}", use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao exibir imagens: {e}")