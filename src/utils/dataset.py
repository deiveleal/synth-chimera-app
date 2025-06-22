
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance # type: ignore
from torchvision.transforms import ToTensor  # type: ignore
from .device_detection import get_available_device
from .image_generation import generate_synthetic_image, SHAPES_BY_CLASS, draw_shape


device = get_available_device()
print(f"Dataset using device: {device}")

# Dataset for generating synthetic multimodal data with images and numerical features
class MultimodalSyntheticDataset(Dataset):
    def __init__(
            self, num_samples=500, 
            num_features=10, 
            image_size=(64, 64), 
            num_classes=2,
            # vary_circle_position=False, 
            # circle_position_max_offset_ratio=0.1,
            # vary_circle_color=False, 
            # circle_color_variation_amount=10,
            # vary_circle_size=False, 
            # circle_size_variation_ratio=0.05,
            num_distractor_objects=3, 
            distractor_max_size_ratio=0.1,
            background_noise_std=0.7,
            apply_blur=True,
            blur_radius_range=(0.5, 1.5),
            apply_brightness_contrast=True,
            brightness_factor_range=(0.7, 1.3),
            contrast_factor_range=(0.7, 1.3),
            enable_multimodal_image_relations=True,
            multimodal_feature_idx=0,
            multimodal_visibility_threshold=0.5
            ):
        super().__init__()
        self.num_samples = num_samples
        self.num_features = num_features
        self.image_size = image_size
        self.num_classes = num_classes

        # self.vary_circle_position = vary_circle_position
        # self.circle_position_max_offset_ratio = circle_position_max_offset_ratio
        # self.vary_circle_color = vary_circle_color
        # self.circle_color_variation_amount = circle_color_variation_amount
        # self.vary_circle_size = vary_circle_size
        # self.circle_size_variation_ratio = circle_size_variation_ratio
        self.num_distractor_objects = num_distractor_objects
        self.distractor_max_size_ratio = distractor_max_size_ratio

        self.background_noise_std = background_noise_std
        self.apply_blur = apply_blur
        self.blur_radius_range = blur_radius_range
        self.apply_brightness_contrast = apply_brightness_contrast
        self.brightness_factor_range = brightness_factor_range
        self.contrast_factor_range = contrast_factor_range
        self.enable_multimodal_image_relations = enable_multimodal_image_relations
        self.multimodal_feature_idx = multimodal_feature_idx
        self.multimodal_visibility_threshold = multimodal_visibility_threshold

        if num_features <= 1: 
            raise ValueError("Number of features must be >1 ")
        
        if num_samples <= 1: 
            raise ValueError("Number of samples must be >1 ")
        
        if num_samples % num_classes != 0: 
            raise ValueError("Number of samples must be divisible by the number of classes")
        
        if num_features % 2 != 0: 
            raise ValueError("The number of features must be an even number")
        if self.enable_multimodal_image_relations and self.multimodal_feature_idx >= num_features:
            raise ValueError("multimodal_feature_idx must be less than num_features")

        self.labels = self._generate_labels()
        self.numerical_features = self._generate_numerical_features()
        self.images = self._generate_images()

    def _generate_labels(self):
        """Generates labels for the dataset."""
        labels_array = np.zeros((self.num_samples), dtype=np.int64)
        samples_per_class = self.num_samples // self.num_classes
        for _class in range(self.num_classes):
            for i in range(samples_per_class):
                sample_index = _class * samples_per_class + i
                labels_array[sample_index] = _class
        return labels_array

    def _generate_numerical_features(self):
        """Generates numerical (structured) features for the dataset."""
        structured_data_array = np.zeros((self.num_samples, self.num_features), dtype=np.float32)
        samples_per_class = self.num_samples // self.num_classes

        for _class in range(self.num_classes):
            ranges_classes = [((_class * (self.num_features // 2) + feature), 
                               (_class * (self.num_features // 2) + feature) + 0.95) 
                              for feature in range(self.num_features // 2)]
            
            for i in range(samples_per_class):
                sample_index = _class * samples_per_class + i

                for feat in range(self.num_features):
                    if feat < (self.num_features // 2):
                        range_idx = feat 
                        range_start, range_finish = ranges_classes[range_idx]
                        structured_data_array[sample_index, feat] = round(np.random.uniform(range_start, range_finish), 2)
                    else:
                        structured_data_array[sample_index, feat] = round(np.random.uniform(100, 100000), 2)
        return structured_data_array

    def _generate_images(self):
        """
        Gera imagens sintéticas usando diferentes formas para cada classe.
        Se a relação multimodal não permitir a forma principal, desenha distratores aleatórios.
        Aplica no máximo um efeito (blur, brilho ou contraste) por imagem.
        """
        if self.num_classes > 7:
            raise ValueError("O número máximo de classes suportado para imagens é 7 (0 a 6).")

        images = []
        for sample_idx, label_value in enumerate(self.labels):
            draw_main_shape = True
            if self.enable_multimodal_image_relations:
                if self.numerical_features[sample_idx, self.multimodal_feature_idx] < self.multimodal_visibility_threshold:
                    draw_main_shape = False

            if draw_main_shape:
                img_np = generate_synthetic_image(
                    class_idx=int(label_value),
                    img_size=self.image_size[0]
                )
                img = Image.fromarray(img_np)
                if self.background_noise_std > 0:
                    img_array = np.array(img)
                    noise = np.random.normal(0, self.background_noise_std * 255, img_array.shape)
                    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
            else:
                img_np = np.full((self.image_size[1], self.image_size[0], 3), 255, dtype=np.uint8)
                n_distractors = np.random.randint(1, 4)
                for _ in range(n_distractors):
                    shape_idx = np.random.randint(0, len(SHAPES_BY_CLASS))
                    shape = SHAPES_BY_CLASS[shape_idx]
                    size = np.random.randint(self.image_size[0] // 8, self.image_size[0] // 3)
                    margin = size + 2
                    center = (
                        np.random.randint(margin, self.image_size[0] - margin),
                        np.random.randint(margin, self.image_size[1] - margin)
                    )
                    color = tuple(np.random.randint(0, 256, 3))
                    draw_shape(img_np, shape, center, size, color)
                img = Image.fromarray(img_np)
                # Não aplica ruído de fundo em imagens só com distratores

            # --- Aplica no máximo UM efeito por imagem ---
            effects = []
            if self.apply_blur:
                effects.append("blur")
            if self.apply_brightness_contrast:
                effects.append("brightness")
                effects.append("contrast")
            effects.append("none")  # opção de não aplicar nada

            chosen_effect = np.random.choice(effects)

            if chosen_effect == "blur":
                blur_radius = np.random.uniform(self.blur_radius_range[0], self.blur_radius_range[1])
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            elif chosen_effect == "brightness":
                brightness_factor = np.random.uniform(self.brightness_factor_range[0], self.brightness_factor_range[1])
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness_factor)
            elif chosen_effect == "contrast":
                contrast_factor = np.random.uniform(self.contrast_factor_range[0], self.contrast_factor_range[1])
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast_factor)
            # Se "none", não faz nada

            images.append(ToTensor()(img).numpy())

        return np.array(images, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.numerical_features[idx], device=device),
            torch.tensor(self.images[idx], device=device),
            torch.tensor(self.labels[idx], device=device),
        )

