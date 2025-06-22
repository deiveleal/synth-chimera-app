import streamlit as st
import pandas as pd # type: ignore
import numpy as np

class DatasetGenerationPage:
    def __init__(self, X_num, X_img, y, n_samples, n_features, 
                 X_num_train=None, X_img_train=None, y_train=None,
                 X_num_val=None, X_img_val=None, y_val=None):
        
        st.title("Dataset Generation")
        
        # Seletor para escolher qual conjunto visualizar
        if X_num_train is not None and X_num_val is not None:
            dataset_view = st.selectbox(
                "Visualizar conjunto de dados:",
                ["Dados Completos", "Dados de Treino", "Dados de Validação"]
            )
            
            if dataset_view == "Dados de Treino":
                X_num, X_img, y = X_num_train, X_img_train, y_train
                n_samples = len(y_train) if y_train is not None else 0  # <-- CORREÇÃO
                st.info(f"Visualizando dados de treino: {n_samples} amostras")
            elif dataset_view == "Dados de Validação":
                X_num, X_img, y = X_num_val, X_img_val, y_val
                n_samples = len(y_val) if y_val is not None else 0      # <-- CORREÇÃO
                st.info(f"Visualizando dados de validação: {n_samples} amostras")
            else:
                st.info(f"Visualizando dados completos: {n_samples} amostras")
        
        # Verificar se dados são válidos antes de processar
        if X_num is None or y is None:
            st.error("Dados não disponíveis para visualização.")
            return
        
        # Conversão mais robusta para numpy
        try:
            X_num_np = X_num.cpu().numpy() if hasattr(X_num, 'cpu') else np.array(X_num)
            y_np = y.cpu().numpy() if hasattr(y, 'cpu') else np.array(y)
        except Exception as e:
            st.error(f"Erro ao converter dados: {e}")
            return
        
        # Verificar dimensões
        if len(X_num_np.shape) != 2:
            st.error("Dados numéricos devem ter 2 dimensões (amostras x features)")
            return
        
        # Usar dimensões reais dos dados
        actual_samples, actual_features = X_num_np.shape
        
        feature_names = [f"Feature_{i+1}" for i in range(actual_features)]  # <-- CORREÇÃO
        df_numeric = pd.DataFrame(X_num_np, columns=feature_names)
        df_numeric['Class'] = y_np
        
        st.subheader("Dataset Information")
        st.write(f"Number of samples: {actual_samples}")                    # <-- CORREÇÃO
        st.write(f"Number of numeric features: {actual_features}")          # <-- CORREÇÃO
        
        # Adicionar estatísticas por classe
        unique_classes, counts = np.unique(y_np, return_counts=True)
        st.write("Class distribution:")
        for class_val, count in zip(unique_classes, counts):
            st.write(f"  - Class {class_val}: {count} samples ({count/len(y_np)*100:.1f}%)")
        
        st.subheader("Numeric Data Sample")
        st.dataframe(df_numeric.head(10))
        
        st.subheader("Image Data Sample")
        if X_img is not None:
            self.display_image_samples(X_img)
        else:
            st.write("Nenhum dado de imagem disponível.")
        
        with st.expander("🔧 Informações de Debug"):
            st.write("**Tipos de dados:**")
            st.write(f"- X_num: {type(X_num)}")
            st.write(f"- X_img: {type(X_img)}")
            st.write(f"- y: {type(y)}")
            
            st.write("**Shapes:**")
            st.write(f"- X_num shape: {X_num.shape if hasattr(X_num, 'shape') else 'N/A'}")
            st.write(f"- X_img shape: {X_img.shape if X_img is not None and hasattr(X_img, 'shape') else 'N/A'}")
            st.write(f"- y shape: {y.shape if hasattr(y, 'shape') else 'N/A'}")
    
    def display_image_samples(self, X_img):
        try:
            # Verificar se X_img é válido
            if X_img is None:
                st.write("Dados de imagem não disponíveis")
                return
            
            # Converter para numpy se necessário
            if hasattr(X_img, 'cpu'):
                X_img_np = X_img.cpu().numpy()
            else:
                X_img_np = np.array(X_img)
            
            if len(X_img_np) == 0:
                st.write("Nenhuma imagem para exibir")
                return
            
            num_samples = min(5, len(X_img_np))
            cols = st.columns(num_samples)
            
            for i in range(num_samples):
                with cols[i]:
                    img = X_img_np[i]
                    
                    # Normalizar valores se necessário
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    # Ajustar dimensões para exibição
                    if len(img.shape) == 3:
                        if img.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
                            img = img.transpose(1, 2, 0)
                        
                        if img.shape[-1] == 1:  # Remover canal único
                            img = img.squeeze(-1)
                    
                    st.image(img, caption=f"Image {i+1}", use_container_width=True)
                    
        except Exception as e:
            st.error(f"Erro ao exibir imagens: {e}")
            st.write("Detalhes do erro para debug:")
            if X_img is not None:
                try:
                    shape_info = X_img.shape if hasattr(X_img, 'shape') else "Shape não disponível"
                    st.write(f"Shape dos dados de imagem: {shape_info}")
                except:
                    st.write("Não foi possível obter informações sobre as imagens")