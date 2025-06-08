import streamlit as st
from utils.device_detection import get_available_device
from utils.dataset import generate_multimodal_dataset


def sidebar():
    st.sidebar.title("Configuração")

    ### Dataset Generation ###
    # Detect device
    device = get_available_device()
    st.sidebar.write(f"Usando dispositivo: {device}")

    # User inputs
    n_samples = st.sidebar.number_input("Número de Amostras:", min_value=1, value=100)
    n_features = st.sidebar.number_input("Número de Características (Número par - Exceto Imagem):", min_value=2, value=10)
    n_classes = st.sidebar.number_input("Número de Classes(Número de amostras deve ser divisível):", min_value=1, value=2)

    if st.sidebar.button("Gerar Dataset"):
        if n_samples > 0 and n_features > 0 and n_classes > 0:
            X_num, X_img, y = generate_multimodal_dataset(
                num_samples=n_samples,
                num_features=n_features,
                num_classes=n_classes
                )
            X_num, X_img, y = X_num.to(device), X_img.to(device), y.to(device)
        else:
            st.error("Por favor, preencha todos os parâmetros corretamente.")

    return None