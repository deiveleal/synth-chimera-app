# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from models.cnn_model import MultimodalCNN


# EVALUATION_CACHE: dict = {}

# def clear_evaluation_cache():
#     """Limpa o cache de avaliação de fitness."""
#     global EVALUATION_CACHE
#     EVALUATION_CACHE.clear()

# def evaluate_features(
#     individual_mask: np.ndarray,
#     X_num_train: torch.Tensor,      
#     X_img_train: torch.Tensor,      
#     y_train: torch.Tensor,          
#     X_num_val: torch.Tensor,        
#     X_img_val: torch.Tensor,        
#     y_val: torch.Tensor,            
#     device: str,
#     epochs: int,
#     algorithm_id: str
# ):
#     global EVALUATION_CACHE

#     mask_tuple = tuple(individual_mask.flatten()) if individual_mask is not None else None
#     use_image_for_cache_key = X_img_train is not None

#     cache_key = (
#         algorithm_id,
#         mask_tuple, 
#         use_image_for_cache_key, 
#         X_num_train.shape[1] if X_num_train is not None else 0,
#         epochs, 
#         device, 
#         str(y_train.shape),
#         str(y_val.shape)  # <-- NOVO: incluir shape de validação
#     )

#     if cache_key in EVALUATION_CACHE:
#         return EVALUATION_CACHE[cache_key]

#     num_selected_numerical_features = 0
#     X_selected_num_train_dev = None
#     X_selected_num_val_dev = None
    
#     if X_num_train is not None and individual_mask is not None:
#         num_selected_numerical_features = int(np.sum(individual_mask))
#         if num_selected_numerical_features > 0:
#             selected_indices = np.where(individual_mask == 1)[0]
#             if selected_indices.size == 0:
#                  EVALUATION_CACHE[cache_key] = -1.0
#                  return -1.0
#             max_index = int(selected_indices.max())
#             if max_index >= X_num_train.shape[1]: 
#                 EVALUATION_CACHE[cache_key] = -1.0 
#                 return -1.0
#             # Aplicar máscara em treino e validação
#             X_selected_num_train_dev = X_num_train[:, selected_indices].to(device)
#             X_selected_num_val_dev = X_num_val[:, selected_indices].to(device)

#     elif X_num_train is None:
#         num_selected_numerical_features = 0
#         X_selected_num_train_dev = None
#         X_selected_num_val_dev = None

#     X_img_train_dev = None
#     X_img_val_dev = None
#     image_input_shape_for_model = None
#     use_image_for_model = False
    
#     if X_img_train is not None:
#         X_img_train_dev = X_img_train.to(device)
#         X_img_val_dev = X_img_val.to(device)
#         image_input_shape_for_model = (X_img_train_dev.shape[1], X_img_train_dev.shape[2], X_img_train_dev.shape[3])
#         use_image_for_model = True

#     if num_selected_numerical_features == 0 and not use_image_for_model:
#         EVALUATION_CACHE[cache_key] = -1.0
#         return -1.0 

#     y_train_dev = y_train.to(device)
#     y_val_dev = y_val.to(device)
    
#     unique_classes = torch.unique(y_train_dev)
#     num_classes = len(unique_classes)

#     if y_train_dev.numel() == 0 or y_val_dev.numel() == 0:
#         EVALUATION_CACHE[cache_key] = -1.0
#         return -1.0
#     if num_classes <= 1: 
#         EVALUATION_CACHE[cache_key] = -1.0 
#         return -1.0

#     try:
#         model = MultimodalCNN(
#             num_struct_features=num_selected_numerical_features,
#             image_input_shape=image_input_shape_for_model,
#             num_classes=num_classes,
#             use_image=use_image_for_model
#         ).to(device)
#     except ValueError as e:
#         EVALUATION_CACHE[cache_key] = -1.0
#         return -1.0 
#     except Exception as e: 
#         EVALUATION_CACHE[cache_key] = -1.0
#         return -1.0

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     for epoch_num_train in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         try:
#             outputs_train = model(structured_data=X_selected_num_train_dev, image_data=X_img_train_dev)
#             loss = criterion(outputs_train, y_train_dev)
#             loss.backward()
#             optimizer.step()
#         except Exception as e:
#             EVALUATION_CACHE[cache_key] = -1.0
#             return -1.0

#     model.eval()
#     accuracy = 0.0
#     try:
#         with torch.no_grad():
#             outputs_eval = model(structured_data=X_selected_num_val_dev, image_data=X_img_val_dev)
#             _, predicted = torch.max(outputs_eval.data, 1)
#             total = y_val_dev.size(0)
#             if total == 0:
#                 EVALUATION_CACHE[cache_key] = -1.0
#                 return -1.0
#             correct = (predicted == y_val_dev).sum().item()
#             accuracy = correct / total  # <-- ACURÁCIA DE VALIDAÇÃO
#     except Exception as e:
#         EVALUATION_CACHE[cache_key] = -1.0
#         return -1.0
    
#     EVALUATION_CACHE[cache_key] = accuracy
#     return accuracy  # <-- Retorna acurácia de validação
#     return accuracy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.cnn_model import MultimodalCNN

EVALUATION_CACHE: dict = {}

def clear_evaluation_cache():
    """Limpa o cache de avaliação de fitness."""
    global EVALUATION_CACHE
    EVALUATION_CACHE.clear()

def evaluate_features(
    individual_mask: np.ndarray,
    X_num_train: torch.Tensor,      
    X_img_train: torch.Tensor,      
    y_train: torch.Tensor,          
    X_num_val: torch.Tensor,        
    X_img_val: torch.Tensor,        
    y_val: torch.Tensor,            
    device: str,
    epochs: int,
    algorithm_id: str
):
    global EVALUATION_CACHE

    # Validação básica de entrada
    if y_train is None or y_val is None:
        return -1.0
    
    if len(y_train) == 0 or len(y_val) == 0:
        return -1.0
    
    # Verificar se ao menos um tipo de dado está disponível
    has_num_data = X_num_train is not None and X_num_val is not None
    has_img_data = X_img_train is not None and X_img_val is not None
    
    if not has_num_data and not has_img_data:
        return -1.0

    # Verificar compatibilidade de dimensões entre treino e validação
    if X_num_train is not None and X_num_val is not None:
        if X_num_train.shape[1] != X_num_val.shape[1]:
            return -1.0
    
    if X_img_train is not None and X_img_val is not None:
        if (X_img_train.shape[1:] != X_img_val.shape[1:]):  # Verificar canais, altura, largura
            return -1.0

    use_image_for_cache_key = X_img_train is not None and X_img_val is not None
    mask_tuple = tuple(individual_mask) if individual_mask is not None else ()

    cache_key = (
        algorithm_id,
        mask_tuple, 
        use_image_for_cache_key, 
        X_num_train.shape[1] if X_num_train is not None else 0,
        epochs, 
        device, 
        str(y_train.shape),
        str(y_val.shape)  # Incluir shape de validação
    )

    if cache_key in EVALUATION_CACHE:
        return EVALUATION_CACHE[cache_key]

    num_selected_numerical_features = 0
    X_selected_num_train_dev = None
    X_selected_num_val_dev = None
    
    if X_num_train is not None and individual_mask is not None:
        num_selected_numerical_features = int(np.sum(individual_mask))
        if num_selected_numerical_features > 0:
            selected_indices = np.where(individual_mask == 1)[0]
            if selected_indices.size == 0:
                 EVALUATION_CACHE[cache_key] = -1.0
                 return -1.0
            max_index = int(selected_indices.max())
            if max_index >= X_num_train.shape[1]: 
                EVALUATION_CACHE[cache_key] = -1.0 
                return -1.0
            # Verificar se X_num_val tem as mesmas colunas
            if max_index >= X_num_val.shape[1]:
                EVALUATION_CACHE[cache_key] = -1.0
                return -1.0
            # Aplicar máscara em treino e validação
            X_selected_num_train_dev = X_num_train[:, selected_indices].to(device)
            X_selected_num_val_dev = X_num_val[:, selected_indices].to(device)

    elif X_num_train is None:
        num_selected_numerical_features = 0
        X_selected_num_train_dev = None
        X_selected_num_val_dev = None

    X_img_train_dev = None
    X_img_val_dev = None
    image_input_shape_for_model = None
    use_image_for_model = False
    
    if X_img_train is not None:
        X_img_train_dev = X_img_train.to(device)
        X_img_val_dev = X_img_val.to(device)
        image_input_shape_for_model = (X_img_train_dev.shape[1], X_img_train_dev.shape[2], X_img_train_dev.shape[3])
        use_image_for_model = True

    y_train_dev = y_train.to(device)
    y_val_dev = y_val.to(device)
    
    unique_classes = torch.unique(y_train_dev)
    num_classes = len(unique_classes)

    if y_train_dev.numel() == 0 or y_val_dev.numel() == 0:
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0
    if num_classes <= 1: 
        EVALUATION_CACHE[cache_key] = -1.0 
        return -1.0

    if X_selected_num_train_dev is None and X_img_train_dev is None:
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0

    try:
        model = MultimodalCNN(
            num_struct_features=num_selected_numerical_features,
            image_input_shape=image_input_shape_for_model,
            num_classes=num_classes,
            use_image=use_image_for_model
        ).to(device)
    except Exception as e:
        print(f"Erro ao criar modelo: {e}")
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TREINAMENTO com dados de treino
    for epoch_num_train in range(epochs):
        model.train()
        optimizer.zero_grad()
        try:
            outputs_train = model(structured_data=X_selected_num_train_dev, image_data=X_img_train_dev)
            loss = criterion(outputs_train, y_train_dev)
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            print(f"Erro durante treinamento: {e}")
            EVALUATION_CACHE[cache_key] = -1.0
            return -1.0
        except Exception as e:
            print(f"Erro inesperado durante treinamento: {e}")
            EVALUATION_CACHE[cache_key] = -1.0
            return -1.0

    # AVALIAÇÃO com dados de validação (FITNESS)
    model.eval()
    accuracy = 0.0
    try:
        with torch.no_grad():
            outputs_eval = model(structured_data=X_selected_num_val_dev, image_data=X_img_val_dev)
            _, predicted = torch.max(outputs_eval.data, 1)
            total = y_val_dev.size(0)
            if total == 0:
                EVALUATION_CACHE[cache_key] = -1.0
                return -1.0
            correct = (predicted == y_val_dev).sum().item()
            accuracy = correct / total
    except RuntimeError as e:
        print(f"Erro durante avaliação: {e}")
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0
    except Exception as e:
        print(f"Erro inesperado durante avaliação: {e}")
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0
    
    EVALUATION_CACHE[cache_key] = accuracy
    return accuracy  # Retorna acurácia de validação