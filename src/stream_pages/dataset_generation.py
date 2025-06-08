# import streamlit as st
# import torch
# import numpy as np
# import fireducks.pandas as pd     # type: ignore
# import matplotlib.pyplot as plt
# import seaborn as sns    # type: ignore
# from utils.generate_dataset import generate_multimodal_dataset
# from utils.device_detection import get_available_device


# class DatasetGenerationPage:
#     def __init__(self, X_num, X_img, y, n_samples, n_features):
#         """
#         Dataset Generation Page for Streamlit App.
#         Args:
#             X_num (torch.Tensor): Numeric features.
#             X_img (torch.Tensor): Image data.
#             y (torch.Tensor): Labels.
#         """
#         st.title("Dataset Multimodal Created by Synth-Chimera")
#         # device = get_available_device()
#         # # Detect device
#         # device = get_available_device()
#         # st.sidebar.write(f"Usando dispositivo: {device}")

#         # # User inputs
#         # n_samples = st.sidebar.number_input("Número de Amostras:", min_value=1, value=100)
#         # n_features = st.sidebar.number_input("Número de Características (Número par - Exceto Imagem):", min_value=2, value=10)
#         # n_classes = st.sidebar.number_input("Número de Classes:", min_value=1, value=2)

#         with st.spinner("Dataset multimodal criado!"):
#             # X_num, X_img, y = generate_multimodal_dataset(num_samples=n_samples, num_features=n_features, image_size=image_size, num_classes=n_classes)
#             # X_num, X_img, y = X_num.to(device), X_img.to(device), y.to(device)

#             st.success("Dataset gerado com sucesso!")
#             st.write("Amostras numéricas:", X_num.shape)
#             st.write("Imagens:", X_img.shape)
#             st.write("Classes:", y.shape)

#             # Salvar os dados na sessão para uso posterior
#             st.session_state.X_num = X_num
#             st.session_state.X_img = X_img
#             st.session_state.y = y

#             # Visualização dos dados numéricos
#             st.subheader("Visualização dos Dados Numéricos")

#             # Converter tensores para dataframe
#             df_numeric = pd.DataFrame(X_num.cpu().numpy())
#             df_numeric.columns = [f"Feature {i+1}" for i in range(n_features)]
#             df_numeric["Class"] = y.cpu().numpy()

#             # Mostrar tabela com os primeiros registros
#             st.write("Primeiras linhas do dataset:")
#             st.dataframe(df_numeric.head(10))

#             # Estatísticas descritivas
#             with st.expander("Estatísticas Descritivas"):
#                 st.write(df_numeric.describe())

#             # Gráficos
#             with st.expander("Visualizações"):
#                 col1, col2 = st.columns(2)

#                 # Histograma de distribuição das características
#                 with col1:
#                     st.write("Distribuição das primeiras características")
#                     fig, ax = plt.subplots(figsize=(10, 6))
#                     cols_to_plot = min(5, n_features)
#                     for i in range(cols_to_plot):
#                         sns.kdeplot(df_numeric[f"Feature {i+1}"], ax=ax, label=f"Feature {i+1}")
#                     plt.legend()
#                     st.pyplot(fig)

#                 # Correlação entre características
#                 with col2:
#                     st.write("Correlação entre características")
#                     fig, ax = plt.subplots(figsize=(10, 8))
#                     # Limitar para as primeiras 10 características para melhor visualização
#                     corr_features = min(10, n_features)
#                     corr_matrix = df_numeric.iloc[:, :corr_features].corr()
#                     sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
#                     st.pyplot(fig)

#                 # Visualização das imagens
#                 st.subheader("Visualização das Imagens")

#                 # Converter imagens para numpy para visualização
#                 images = X_img.cpu().numpy()
#                 labels = y.cpu().numpy()

#                 # Parâmetros para a grade de imagens
#                 n_classes_unique = len(np.unique(labels))
#                 images_per_class = min(5, n_samples // n_classes_unique)

#                 # Criar uma grade de imagens
#                 fig, axes = plt.subplots(n_classes_unique, images_per_class, 
#                                         figsize=(images_per_class*2, n_classes_unique*2))

#                 # Ajustar layout para uma única classe
#                 if n_classes_unique == 1:
#                     axes = np.array([axes])

#                 for class_idx in range(n_classes_unique):
#                     class_indices = np.where(labels == class_idx)[0]
#                     selected_indices = class_indices[:images_per_class]

#                     for img_idx, sample_idx in enumerate(selected_indices):
#                         if len(selected_indices) > 0:
#                             if n_classes_unique == 1:
#                                 ax = axes[img_idx]
#                             else:
#                                 ax = axes[class_idx, img_idx]

#                             # Pegar a imagem e normalizar para visualização
#                             img = images[sample_idx].transpose(1, 2, 0)  # Mover canais para o final (H, W, C)

#                             # Verificar se a imagem é grayscale ou RGB
#                             if img.shape[2] == 1:
#                                 img = img.squeeze()
#                                 ax.imshow(img, cmap='gray')
#                             else:
#                                 # Normalizar imagem para exibição
#                                 img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#                                 ax.imshow(img)

#                             ax.set_title(f"Classe {class_idx}")
#                             ax.axis('off')

#                 plt.tight_layout()
#                 st.pyplot(fig)


# if __name__ == "__main__":
#     DatasetGenerationPage()
import streamlit as st
import pandas as pd
import numpy as np

class DatasetGenerationPage:
    def __init__(self, X_num, X_img, y, n_samples, n_features):
        st.title("Dataset Generation")
        
        # Converter tensores PyTorch para numpy antes de criar DataFrames
        X_num_np = X_num.cpu().numpy() if hasattr(X_num, 'cpu') else np.array(X_num)
        y_np = y.cpu().numpy() if hasattr(y, 'cpu') else np.array(y)
        
        # Criar DataFrame com os dados convertidos
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        df_numeric = pd.DataFrame(X_num_np, columns=feature_names)
        df_numeric['Class'] = y_np
        
        # Mostrar informações do dataset
        st.subheader("Dataset Information")
        st.write(f"Number of samples: {n_samples}")
        st.write(f"Number of numeric features: {n_features}")
        
        # Exibir amostras dos dados
        st.subheader("Numeric Data Sample")
        st.dataframe(df_numeric.head(10))
        
        # Visualizar algumas imagens
        st.subheader("Image Data Sample")
        if hasattr(X_img, 'cpu'):
            # Converter e mostrar algumas imagens
            self.display_image_samples(X_img)
    
    def display_image_samples(self, X_img):
        try:
            # Mostrar até 5 imagens de exemplo
            num_samples = min(5, len(X_img))
            cols = st.columns(num_samples)
            
            for i in range(num_samples):
                with cols[i]:
                    # Converter tensor para numpy para exibição
                    img = X_img[i].cpu().numpy()
                    
                    # Ajustar dimensões e normalizar se necessário
                    if len(img.shape) > 2:
                        if img.shape[0] in (1, 3):  # Formato CHW
                            img = img.transpose(1, 2, 0)
                        
                        if img.shape[-1] == 1:  # Imagem em escala de cinza
                            img = img.squeeze()
                    
                    st.image(img, caption=f"Image {i+1}", use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao exibir imagens: {e}")