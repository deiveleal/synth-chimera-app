# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from .cnn_model import MultimodalCNN

# # def evaluate_features(X_num, X_img, y, device, use_image=True, num_epochs=20, lr=0.001):
# #     """
# #     Evaluate feature selection using MultimodalCNN as the fitness function.

# #     Args:
# #         X_num (torch.Tensor): Structured data.
# #         X_img (torch.Tensor): Image data.
# #         y (torch.Tensor): Labels.
# #         device (str): Device to run the model on.
# #         use_image (bool): If True, use image data; otherwise, ignore image data.
# #         num_epochs (int): Number of training epochs.
# #         lr (float): Learning rate.

# #     Returns:
# #         float: Final model accuracy on the dataset.
# #     """
# #     print(f"CNN using device: {device}")
# #     # Ensure inputs are tensors
# #     num_struct_features = X_num.shape[1]
# #     image_input_shape = X_img.shape[1:]  # C, H, W
# #     num_classes = len(torch.unique(y))

# #     # Initialize model
# #     model = MultimodalCNN(num_struct_features, image_input_shape, num_classes, use_image=use_image).to(device)
# #     criterion = nn.CrossEntropyLoss()
# #     optimizer = optim.Adam(model.parameters(), lr=lr)

# #     # Training loop
# #     model.train()
# #     for epoch in range(num_epochs):
# #         optimizer.zero_grad()
# #         if use_image:
# #             outputs = model(X_num, X_img)
# #         else:
# #             outputs = model(X_num)  # Apenas os dados estruturados
# #         loss = criterion(outputs, y)
# #         loss.backward()
# #         optimizer.step()

# #     # Calculate accuracy
# #     model.eval()
# #     with torch.no_grad():
# #         if use_image:
# #             outputs = model(X_num, X_img)
# #         else:
# #             outputs = model(X_num)  # Apenas os dados estruturados
# #         _, predicted = torch.max(outputs, 1)
# #         accuracy = (predicted == y).float().mean().item()

# #     del model

# #     return accuracy

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from .cnn_model import MultimodalCNN # Certifique-se que os modelos estão corretos e importados

# def evaluate_features(
#     individual_mask: np.ndarray, # Máscara binária numpy
#     X_num_original: torch.Tensor, # Tensor PyTorch com TODAS as features numéricas
#     X_img_tensor: torch.Tensor,
#     y_tensor: torch.Tensor,
#     use_image: bool,
#     model_architecture: str, # ex: "SimpleCNN"
#     device: str,
#     epochs: int = 5 # Número de épocas para treinar a CNN em cada avaliação
# ):
#     """
#     Avalia um subconjunto de features usando uma CNN.
#     Retorna o valor de fitness (menor é melhor, ex: loss).
#     """
#     num_selected_numerical_features = np.sum(individual_mask)

#     if num_selected_numerical_features == 0 and not use_image:
#         # print("DEBUG evaluate_features: No features selected and no image. Returning inf.")
#         return float('inf')

#     X_selected_num_tensor = None
#     if num_selected_numerical_features > 0:
#         selected_indices = np.where(individual_mask == 1)[0]
#         X_selected_num_tensor = X_num_original[:, selected_indices].to(device)
    
#     # Mover outros tensores para o dispositivo (já devem estar, mas para garantir)
#     X_img_tensor_dev = X_img_tensor.to(device) if X_img_tensor is not None else None
#     y_tensor_dev = y_tensor.to(device)

#     num_classes = len(torch.unique(y_tensor_dev))
#     model = None
#     input_for_model = None

#     if model_architecture == "SimpleCNN": # Este nome pode ser genérico
#         if use_image and X_selected_num_tensor is not None: # Combinado
#             model = CombinedModel(
#                 num_features=X_selected_num_tensor.shape[1],
#                 img_channels=X_img_tensor_dev.shape[1],
#                 img_height=X_img_tensor_dev.shape[2],
#                 img_width=X_img_tensor_dev.shape[3],
#                 num_classes=num_classes
#             ).to(device)
#             input_for_model = (X_selected_num_tensor, X_img_tensor_dev)
#         elif use_image: # Apenas Imagem
#             model = CombinedModel( # Usando CombinedModel com num_features=0
#                 num_features=0, # Importante
#                 img_channels=X_img_tensor_dev.shape[1],
#                 img_height=X_img_tensor_dev.shape[2],
#                 img_width=X_img_tensor_dev.shape[3],
#                 num_classes=num_classes,
#                 use_image_only=True # Adicionar este flag ao seu CombinedModel
#             ).to(device)
#             input_for_model = (None, X_img_tensor_dev) # Passar None para x_num
#         elif X_selected_num_tensor is not None: # Apenas Numérico
#             model = SimpleCNN(
#                 input_dim=X_selected_num_tensor.shape[1],
#                 num_classes=num_classes
#             ).to(device)
#             input_for_model = X_selected_num_tensor
#         else: # Nenhuma feature e use_image é False (já tratado no início)
#             return float('inf') 
#     else:
#         raise ValueError(f"Model architecture '{model_architecture}' not supported in cnn_fitness.")

#     if model is None: # Segurança
#         # print("DEBUG evaluate_features: Model is None. Returning inf.")
#         return float('inf')

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001) # lr pode ser um parâmetro

#     model.train()
#     final_loss = float('inf')
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         if isinstance(input_for_model, tuple):
#             outputs = model(*input_for_model)
#         else:
#             outputs = model(input_for_model)
#         loss = criterion(outputs, y_tensor_dev)
#         loss.backward()
#         optimizer.step()
#         if epoch == epochs -1 : # Pegar a loss da última época
#             final_loss = loss.item()
    
#     # print(f"DEBUG evaluate_features: Mask sum {num_selected_numerical_features}, UseImg: {use_image}, Loss: {final_loss:.4f}")
#     return final_loss

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from .cnn_model import MultimodalCNN # Importar o modelo fornecido

# def evaluate_features(
#     individual_mask: np.ndarray,    # Máscara binária numpy para features numéricas (pode ser None se não houver features numéricas)
#     X_num_original: torch.Tensor,   # Tensor PyTorch com TODAS as features numéricas (pode ser None)
#     X_img_tensor: torch.Tensor,     # Tensor PyTorch com dados de imagem (pode ser None)
#     y_tensor: torch.Tensor,
#     device: str,
#     epochs: int = 5 # Número de épocas para treinar a CNN em cada avaliação
# ):
#     """
#     Avalia um subconjunto de features usando a MultimodalCNN.
#     Retorna o valor de fitness (menor é melhor, ex: loss).
#     """
#     num_selected_numerical_features = 0
#     X_selected_num_tensor_dev = None # Dados numéricos selecionados para o modelo
    
#     # Lógica para dados numéricos
#     if X_num_original is not None:
#         if individual_mask is not None: # Se há máscara, aplicar
#             num_selected_numerical_features = int(np.sum(individual_mask))
#             if num_selected_numerical_features > 0:
#                 selected_indices = np.where(individual_mask == 1)[0]
#                 if X_num_original.shape[1] < int(max(selected_indices)) + 1:
#                     print(f"ERRO: Máscara de features excede dimensões de X_num_original. Mask len: {len(individual_mask)}, X_num_original shape: {X_num_original.shape}")
#                     return float('inf')
#                 X_selected_num_tensor_dev = X_num_original[:, selected_indices].to(device)
#             # Se num_selected_numerical_features é 0 após a máscara, X_selected_num_tensor_dev permanece None
#         else: # Sem máscara, usar todas as features numéricas originais
#             num_selected_numerical_features = X_num_original.shape[1]
#             X_selected_num_tensor_dev = X_num_original.to(device)
#     # Se X_num_original é None, num_selected_numerical_features permanece 0 e X_selected_num_tensor_dev é None

#     # Lógica para dados de imagem
#     X_img_tensor_dev = None
#     image_input_shape_for_model = None # (C, H, W)
#     use_image_for_model = False
#     if X_img_tensor is not None:
#         X_img_tensor_dev = X_img_tensor.to(device)
#         image_input_shape_for_model = (X_img_tensor_dev.shape[1], X_img_tensor_dev.shape[2], X_img_tensor_dev.shape[3])
#         use_image_for_model = True # O modelo decidirá internamente com base neste e no seu próprio 'use_image'

#     # Condição de parada se nenhuma modalidade estiver ativa
#     if num_selected_numerical_features == 0 and not use_image_for_model:
#         # print("DEBUG evaluate_features: No numerical features and no image data. Returning inf.")
#         return float('inf')

#     y_tensor_dev = y_tensor.to(device)
#     num_classes = len(torch.unique(y_tensor_dev))

#     try:
#         # num_struct_features é o número de features que o ramo estruturado do modelo espera.
#         # image_input_shape é o formato da imagem que o ramo de imagem espera.
#         # use_image controla se o ramo de imagem é construído/usado.
#         model = MultimodalCNN(
#             num_struct_features=num_selected_numerical_features, # Número de features numéricas efetivamente passadas
#             image_input_shape=image_input_shape_for_model, # Pode ser None
#             num_classes=num_classes,
#             use_image=use_image_for_model # Controla a construção do ramo de imagem no modelo
#         ).to(device)
#     except ValueError as e:
#         print(f"Erro ao instanciar MultimodalCNN: {e}. NumStructFeat: {num_selected_numerical_features}, ImgShape: {image_input_shape_for_model}, UseImg: {use_image_for_model}")
#         return float('inf')

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     model.train()
#     final_loss = float('inf')

#     # O forward do modelo espera structured_data e image_data.
#     # Passe None se a respectiva modalidade não estiver sendo usada.
#     # A lógica interna do modelo (baseada em num_struct_features > 0 e self.use_image)
#     # determinará se os dados None são um problema.
    
#     # structured_data para o forward é X_selected_num_tensor_dev
#     # image_data para o forward é X_img_tensor_dev

#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         try:
#             # Passe os dados para o modelo; o modelo internamente sabe se deve usá-los
#             outputs = model(structured_data=X_selected_num_tensor_dev, image_data=X_img_tensor_dev)
#         except ValueError as e:
#             print(f"Erro durante o forward pass do modelo: {e}")
#             # print(f"DEBUG forward error: X_selected_num_tensor_dev is None: {X_selected_num_tensor_dev is None}, X_img_tensor_dev is None: {X_img_tensor_dev is None}")
#             # if X_selected_num_tensor_dev is not None: print(f"DEBUG forward error: X_selected_num_tensor_dev shape: {X_selected_num_tensor_dev.shape}")
#             return float('inf')

#         loss = criterion(outputs, y_tensor_dev)
#         loss.backward()
#         optimizer.step()
#         if epoch == epochs - 1:
#             final_loss = loss.item()
    
#     # print(f"DEBUG evaluate_features: NumSelectFeat: {num_selected_numerical_features}, ImgShape: {image_input_shape_for_model}, UseImgForModel: {use_image_for_model}, Loss: {final_loss:.4f}")
#     return final_loss

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from .cnn_model import MultimodalCNN # Ou o nome correto do seu modelo

# def evaluate_features(
#     individual_mask: np.ndarray,
#     X_num_original: torch.Tensor,
#     X_img_tensor: torch.Tensor, # Pode ser None
#     y_tensor: torch.Tensor,
#     device: str,
#     epochs: int = 5
# ):
#     num_selected_numerical_features = 0
#     X_selected_num_tensor_dev = None
    
#     if X_num_original is not None and individual_mask is not None:
#         num_selected_numerical_features = int(np.sum(individual_mask))
#         if num_selected_numerical_features > 0:
#             selected_indices = np.where(individual_mask == 1)[0]
#             if X_num_original.shape[1] < max(selected_indices) + 1:
#                 return -1.0 # Retornar fitness ruim (menor possível)
#             X_selected_num_tensor_dev = X_num_original[:, selected_indices].to(device)
#     elif X_num_original is not None and individual_mask is None: # Usar todas as features
#         num_selected_numerical_features = X_num_original.shape[1]
#         X_selected_num_tensor_dev = X_num_original.to(device)

#     X_img_tensor_dev = None
#     image_input_shape_for_model = None
#     use_image_for_model = False
#     if X_img_tensor is not None:
#         X_img_tensor_dev = X_img_tensor.to(device)
#         image_input_shape_for_model = (X_img_tensor_dev.shape[1], X_img_tensor_dev.shape[2], X_img_tensor_dev.shape[3])
#         use_image_for_model = True

#     if num_selected_numerical_features == 0 and not use_image_for_model:
#         return -1.0 # Retornar fitness ruim

#     y_tensor_dev = y_tensor.to(device)
#     num_classes = len(torch.unique(y_tensor_dev))
#     if num_classes <= 1 and y_tensor_dev.numel() > 0: # Evitar erro com CrossEntropyLoss se houver apenas uma classe
#         # Se só há uma classe, a acurácia é trivial ou indefinida dependendo da perspectiva.
#         # Para seleção de features, isso pode não ser um bom cenário.
#         # Retornar um valor que indique isso, ou 1.0 se todas as predições forem para essa classe.
#         # Por simplicidade, se o objetivo é maximizar, um fitness baixo aqui é apropriado.
#         print(f"Warning: Only one class detected in y_tensor. Num_classes: {num_classes}")
#         # Se o modelo prever a única classe corretamente, acurácia é 1.
#         # Mas o treinamento da CNN pode ser problemático.
#         # Vamos retornar 0 como fitness neutro/ruim para evitar problemas com a otimização.
#         return 0.0


#     try:
#         model = MultimodalCNN(
#             num_struct_features=num_selected_numerical_features,
#             image_input_shape=image_input_shape_for_model,
#             num_classes=num_classes,
#             use_image=use_image_for_model
#         ).to(device)
#     except ValueError as e:
#         print(f"Erro ao instanciar MultimodalCNN: {e}")
#         return -1.0 # Fitness ruim

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Loop de treinamento
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         try:
#             outputs_train = model(structured_data=X_selected_num_tensor_dev, image_data=X_img_tensor_dev)
#             loss = criterion(outputs_train, y_tensor_dev)
#             loss.backward()
#             optimizer.step()
#         except ValueError as e:
#             print(f"Erro durante o forward/backward pass do modelo: {e}")
#             return -1.0 # Fitness ruim
#         except RuntimeError as e: # Capturar erros de runtime comuns no treinamento
#             print(f"Runtime error durante o treinamento: {e}")
#             return -1.0

#     # Avaliação da acurácia
#     model.eval()
#     accuracy = 0.0
#     try:
#         with torch.no_grad():
#             outputs_eval = model(structured_data=X_selected_num_tensor_dev, image_data=X_img_tensor_dev)
#             _, predicted_classes = torch.max(outputs_eval, 1)
#             correct_predictions = (predicted_classes == y_tensor_dev).sum().item()
#             total_predictions = y_tensor_dev.size(0)
#             if total_predictions > 0:
#                 accuracy = correct_predictions / total_predictions
#             # else: acurácia permanece 0.0 se não houver predições (y_tensor vazio)
#     except ValueError as e:
#         print(f"Erro durante o forward pass de avaliação: {e}")
#         return -1.0 # Fitness ruim
#     except RuntimeError as e:
#         print(f"Runtime error durante a avaliação: {e}")
#         return -1.0

#     # print(f"DEBUG evaluate_features: Accuracy: {accuracy:.4f}")
#     return accuracy # RETORNA ACURÁCIA DIRETAMENTE (MAIOR É MELHOR)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.cnn_model import MultimodalCNN # Ou o nome correto do seu modelo

def evaluate_features(
    individual_mask: np.ndarray,
    X_num_original: torch.Tensor,
    X_img_tensor: torch.Tensor, # Pode ser None
    y_tensor: torch.Tensor,
    device: str,
    epochs: int = 5
):
    # DEBUG: Print de entrada
    # print(f"DEBUG evaluate_features: Called with individual_mask shape: {individual_mask.shape if individual_mask is not None else 'None'}, "
    #       f"X_num_original shape: {X_num_original.shape if X_num_original is not None else 'None'}, "
    #       f"X_img_tensor shape: {X_img_tensor.shape if X_img_tensor is not None else 'None'}, "
    #       f"y_tensor shape: {y_tensor.shape if y_tensor is not None else 'None'}")
    # print(f"DEBUG evaluate_features: Mask: {individual_mask if individual_mask is not None else 'N/A'}")


    num_selected_numerical_features = 0
    X_selected_num_tensor_dev = None
    
    if X_num_original is not None and individual_mask is not None:
        num_selected_numerical_features = int(np.sum(individual_mask))
        print(f"DEBUG evaluate_features: Num selected numerical features: {num_selected_numerical_features}")
        if num_selected_numerical_features > 0:
            selected_indices = np.where(individual_mask == 1)[0]
            max_index = int(selected_indices.max())
            if X_num_original.shape[1] < max_index + 1:
                print(f"ERROR evaluate_features: Invalid individual_mask. Max index {max_index} out of bounds for X_num_original with {X_num_original.shape[1]} features.")
                return -1.0 # Retornar fitness ruim (menor possível)
            X_selected_num_tensor_dev = X_num_original[:, selected_indices].to(device)
            print(f"DEBUG evaluate_features: X_selected_num_tensor_dev shape: {X_selected_num_tensor_dev.shape}")
    elif X_num_original is not None and individual_mask is None: 
        num_selected_numerical_features = X_num_original.shape[1]
        X_selected_num_tensor_dev = X_num_original.to(device)
        print(f"DEBUG evaluate_features: Using all numerical features: {num_selected_numerical_features}")

    X_img_tensor_dev = None
    image_input_shape_for_model = None
    use_image_for_model = False
    if X_img_tensor is not None:
        X_img_tensor_dev = X_img_tensor.to(device)
        image_input_shape_for_model = (X_img_tensor_dev.shape[1], X_img_tensor_dev.shape[2], X_img_tensor_dev.shape[3])
        use_image_for_model = True
        print(f"DEBUG evaluate_features: Using image data. Shape for model: {image_input_shape_for_model}")
    else:
        print(f"DEBUG evaluate_features: Not using image data.")

    if num_selected_numerical_features == 0 and not use_image_for_model:
        print("DEBUG evaluate_features: No features selected (numerical or image). Returning -1.0")
        return -1.0 

    y_tensor_dev = y_tensor.to(device)
    num_classes = len(torch.unique(y_tensor_dev))
    if num_classes <= 1 and y_tensor_dev.numel() > 0: 
        print(f"DEBUG evaluate_features: Warning - Only one class ({num_classes}) detected in y_tensor. Returning 0.0 fitness.")
        return 0.0

    print(f"DEBUG evaluate_features: Num classes for model: {num_classes}")

    try:
        model = MultimodalCNN(
            num_struct_features=num_selected_numerical_features,
            image_input_shape=image_input_shape_for_model,
            num_classes=num_classes,
            use_image=use_image_for_model
        ).to(device)
        # print(f"DEBUG evaluate_features: Model instantiated successfully.")
    except ValueError as e:
        print(f"ERROR evaluate_features: Failed to instantiate MultimodalCNN: {e}")
        return -1.0 
    except Exception as e: # Captura mais genérica para erros de instanciação
        print(f"ERROR evaluate_features: Unexpected error instantiating MultimodalCNN: {e}")
        return -1.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate pode ser um parâmetro

    # Loop de treinamento
    # print(f"DEBUG evaluate_features: Starting training for {epochs} epochs...")
    for epoch_num in range(epochs):
        model.train()
        optimizer.zero_grad()
        try:
            outputs_train = model(structured_data=X_selected_num_tensor_dev, image_data=X_img_tensor_dev)
            loss = criterion(outputs_train, y_tensor_dev)
            loss.backward()
            optimizer.step()
            # print(f"DEBUG evaluate_features: Epoch {epoch_num+1}/{epochs}, Loss: {loss.item():.4f}")
        except ValueError as e:
            print(f"ERROR evaluate_features: ValueError during model training (epoch {epoch_num+1}): {e}")
            return -1.0 
        except RuntimeError as e: 
            print(f"ERROR evaluate_features: RuntimeError during model training (epoch {epoch_num+1}): {e}")
            return -1.0
        except Exception as e:
            print(f"ERROR evaluate_features: Unexpected error during model training (epoch {epoch_num+1}): {e}")
            return -1.0


    # Avaliação da acurácia
    model.eval()
    accuracy = 0.0
    print(f"DEBUG evaluate_features: Starting evaluation...")
    try:
        with torch.no_grad():
            outputs_eval = model(structured_data=X_selected_num_tensor_dev, image_data=X_img_tensor_dev)
            _, predicted_classes = torch.max(outputs_eval, 1)
            correct_predictions = (predicted_classes == y_tensor_dev).sum().item()
            total_predictions = y_tensor_dev.size(0)
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
            print(f"DEBUG evaluate_features: Evaluation - Correct: {correct_predictions}, Total: {total_predictions}, Accuracy: {accuracy:.4f}")
            
    except ValueError as e:
        print(f"ERROR evaluate_features: ValueError during model evaluation: {e}")
        return -1.0 
    except RuntimeError as e:
        print(f"ERROR evaluate_features: RuntimeError during model evaluation: {e}")
        return -1.0
    except Exception as e:
        print(f"ERROR evaluate_features: Unexpected error during model evaluation: {e}")
        return -1.0

    print(f"DEBUG evaluate_features: Returning final accuracy: {accuracy:.4f}")
    return accuracy