# import torch
# import torch.nn as nn

# class MultimodalCNN(nn.Module):
#     def __init__(self, num_struct_features, image_input_shape, num_classes, use_image=True):
#         super(MultimodalCNN, self).__init__()

#         # Structured Data Branch
#         self.structured_branch = nn.Sequential(
#             nn.Linear(num_struct_features, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )

#         # Image Data Branch (será desativado se use_image for False)
#         self.use_image = use_image
#         if self.use_image:
#             self.image_branch = nn.Sequential(
#                 nn.Conv2d(in_channels=image_input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 2),
#                 nn.Conv2d(16, 32, 3, 1, 1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 2),
#                 nn.Flatten(),
#                 nn.Linear(32 * (image_input_shape[1] // 4) * (image_input_shape[2] // 4), 64),
#                 nn.ReLU()
#             )
#         else:
#             self.image_branch = None

#         # Fusion Layer
#         self.fusion = nn.Sequential(
#             nn.Linear(32 + (64 if self.use_image else 0), 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, structured_data, image_data=None):
#         # Process structured data
#         structured_out = self.structured_branch(structured_data)
        
#         # Process image data (se necessário)
#         if self.use_image and image_data is not None:
#             image_out = self.image_branch(image_data)
#             combined = torch.cat((structured_out, image_out), dim=1)
#         else:
#             combined = structured_out

#         # Final classification
#         output = self.fusion(combined)
#         return output

import torch
import torch.nn as nn

class MultimodalCNN(nn.Module):
    def __init__(self, num_struct_features, image_input_shape, num_classes, use_image=True):
        super(MultimodalCNN, self).__init__()

        # Structured Data Branch
        self.num_struct_features = num_struct_features # Armazenar para referência
        if self.num_struct_features > 0:
            self.structured_branch = nn.Sequential(
                nn.Linear(num_struct_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            self.structured_output_size = 32
        else:
            self.structured_branch = None
            self.structured_output_size = 0


        # Image Data Branch (será desativado se use_image for False ou image_input_shape for None)
        self.use_image = use_image and (image_input_shape is not None)
        self.image_output_size = 0
        if self.use_image:
            if image_input_shape is None or len(image_input_shape) != 3:
                raise ValueError("image_input_shape must be a tuple/list of 3 (channels, height, width) when use_image is True.")
            
            self.image_branch = nn.Sequential(
                nn.Conv2d(in_channels=image_input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # Corrigido kernel_size
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                # A camada linear precisa do tamanho correto após o flatten
                # Se image_input_shape = (C, H, W), após 2 MaxPool2d(2,2), H -> H//4, W -> W//4
                # Então o tamanho achatado é 32 * (H//4) * (W//4)
                nn.Linear(32 * (image_input_shape[1] // 4) * (image_input_shape[2] // 4), 64),
                nn.ReLU()
            )
            self.image_output_size = 64
        else:
            self.image_branch = None
        
        if self.structured_output_size == 0 and self.image_output_size == 0:
            raise ValueError("MultimodalCNN must have at least one active branch (structured or image).")

        # Fusion Layer
        # O tamanho da entrada para a fusão é a soma das saídas dos ramos ativos
        fusion_input_size = self.structured_output_size + self.image_output_size
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, structured_data=None, image_data=None):
        structured_out = None
        image_out = None

        if self.structured_branch is not None:
            if structured_data is None:
                raise ValueError("Structured data expected but not provided.")
            if structured_data.shape[1] != self.num_struct_features:
                raise ValueError(f"Expected {self.num_struct_features} structured features, got {structured_data.shape[1]}")
            structured_out = self.structured_branch(structured_data)
        
        if self.use_image and self.image_branch is not None: # self.use_image já verifica image_input_shape
            if image_data is None:
                raise ValueError("Image data expected but not provided for active image branch.")
            image_out = self.image_branch(image_data)
        
        # Concatenar as saídas dos ramos ativos
        if structured_out is not None and image_out is not None:
            combined = torch.cat((structured_out, image_out), dim=1)
        elif structured_out is not None:
            combined = structured_out
        elif image_out is not None:
            combined = image_out
        else:
            # Esta situação não deve ocorrer se o __init__ validar corretamente
            raise ValueError("No data from any branch to fuse.")

        # Final classification
        output = self.fusion(combined)
        return output