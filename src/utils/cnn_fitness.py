import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import MultimodalCNN


EVALUATION_CACHE: dict = {}

def clear_evaluation_cache():
    """Limpa o cache de avaliação de fitness."""
    global EVALUATION_CACHE
    EVALUATION_CACHE.clear()

def evaluate_features(
    individual_mask: np.ndarray,
    X_num_original: torch.Tensor,
    X_img_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    device: str,
    epochs: int,
    algorithm_id: str
):
    global EVALUATION_CACHE

    mask_tuple = tuple(individual_mask.flatten()) if individual_mask is not None else None
    use_image_for_cache_key = X_img_tensor is not None

    cache_key = (
        algorithm_id,
        mask_tuple, 
        use_image_for_cache_key, 
        X_num_original.shape[1] if X_num_original is not None else 0,
        epochs, 
        device, 
        str(y_tensor.shape)
    )

    if cache_key in EVALUATION_CACHE:
        return EVALUATION_CACHE[cache_key]

    num_selected_numerical_features = 0
    X_selected_num_tensor_dev = None
    
    if X_num_original is not None and individual_mask is not None:
        num_selected_numerical_features = int(np.sum(individual_mask))
        if num_selected_numerical_features > 0:
            selected_indices = np.where(individual_mask == 1)[0]
            if selected_indices.size == 0:
                 EVALUATION_CACHE[cache_key] = -1.0
                 return -1.0
            max_index = int(selected_indices.max())
            if max_index >= X_num_original.shape[1]: 
                EVALUATION_CACHE[cache_key] = -1.0 
                return -1.0
            X_selected_num_tensor_dev = X_num_original[:, selected_indices].to(device)

    elif X_num_original is None:
        num_selected_numerical_features = 0
        X_selected_num_tensor_dev = None

    X_img_tensor_dev = None
    image_input_shape_for_model = None
    use_image_for_model = False
    if X_img_tensor is not None:
        X_img_tensor_dev = X_img_tensor.to(device)
        image_input_shape_for_model = (X_img_tensor_dev.shape[1], X_img_tensor_dev.shape[2], X_img_tensor_dev.shape[3])
        use_image_for_model = True

    if num_selected_numerical_features == 0 and not use_image_for_model:
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0 

    y_tensor_dev = y_tensor.to(device)
    unique_classes = torch.unique(y_tensor_dev)
    num_classes = len(unique_classes)

    if y_tensor_dev.numel() == 0:
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0
    if num_classes <= 1: 
        EVALUATION_CACHE[cache_key] = -1.0 
        return -1.0

    try:
        model = MultimodalCNN(
            num_struct_features=num_selected_numerical_features,
            image_input_shape=image_input_shape_for_model,
            num_classes=num_classes,
            use_image=use_image_for_model
        ).to(device)
    except ValueError as e:
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0 
    except Exception as e: 
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch_num_train in range(epochs):
        model.train()
        optimizer.zero_grad()
        try:
            outputs_train = model(structured_data=X_selected_num_tensor_dev, image_data=X_img_tensor_dev)
            loss = criterion(outputs_train, y_tensor_dev)
            loss.backward()
            optimizer.step()
        except Exception as e:
            EVALUATION_CACHE[cache_key] = -1.0
            return -1.0

    model.eval()
    accuracy = 0.0
    try:
        with torch.no_grad():
            outputs_eval = model(structured_data=X_selected_num_tensor_dev, image_data=X_img_tensor_dev)
            _, predicted = torch.max(outputs_eval.data, 1)
            total = y_tensor_dev.size(0)
            if total == 0:
                EVALUATION_CACHE[cache_key] = -1.0
                return -1.0
            correct = (predicted == y_tensor_dev).sum().item()
            accuracy = correct / total
    except Exception as e:
        EVALUATION_CACHE[cache_key] = -1.0
        return -1.0
    
    EVALUATION_CACHE[cache_key] = accuracy
    return accuracy