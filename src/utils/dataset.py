
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance # type: ignore
from torchvision.transforms import ToTensor  # type: ignore
from .device_detection import get_available_device

device = get_available_device()
print(f"Dataset using device: {device}")

class MultimodalSyntheticDataset(Dataset):
    def __init__(
            self, num_samples=500, 
            num_features=10, 
            image_size=(64, 64), 
            num_classes=2,
            vary_circle_position=False, 
            circle_position_max_offset_ratio=0.1,
            vary_circle_color=False, 
            circle_color_variation_amount=10,
            vary_circle_size=False, 
            circle_size_variation_ratio=0.05,
            num_distractor_objects=3, 
            distractor_max_size_ratio=0.1,
            background_noise_std=0.7, # New: Background noise level (0-1, std dev)
            apply_blur=False, # New: Apply Gaussian blur
            blur_radius_range=(0.5, 1.5), # New: Blur radius range
            apply_brightness_contrast=False, # New: Apply brightness/contrast adjustment
            brightness_factor_range=(0.7, 1.3), # New: Brightness factor range
            contrast_factor_range=(0.7, 1.3), # New: Contrast factor range
            enable_multimodal_image_relations=False, # New: Enable multimodal image relations
            multimodal_feature_idx=0, # New: Index of numerical feature for relation
            multimodal_visibility_threshold=0.5 # New: Threshold for conditional visibility
            ):
        super().__init__()
        self.num_samples = num_samples
        self.num_features = num_features
        self.image_size = image_size
        self.num_classes = num_classes

        # Parameters for image complexity
        self.vary_circle_position = vary_circle_position
        self.circle_position_max_offset_ratio = circle_position_max_offset_ratio
        self.vary_circle_color = vary_circle_color
        self.circle_color_variation_amount = circle_color_variation_amount
        self.vary_circle_size = vary_circle_size
        self.circle_size_variation_ratio = circle_size_variation_ratio
        self.num_distractor_objects = num_distractor_objects
        self.distractor_max_size_ratio = distractor_max_size_ratio

        # Parameters for background noise, augmentations, and multimodal relations
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

        # Generate data components
        self.labels = self._generate_labels() # Generate labels first
        self.numerical_features = self._generate_numerical_features() # Then numerical features (can depend on labels if needed in future)
        self.images = self._generate_images() # Then images (can depend on labels and numerical_features)

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
            # print(f'Generating features for label {_class}') # Optional: for debugging
            # Creating specific random intervals for each features in classes
            # This logic assumes self.labels is already populated if ranges depend on class,
            # but here ranges_classes depends only on _class and num_features.
            ranges_classes = [((_class * (self.num_features // 2) + feature), 
                               (_class * (self.num_features // 2) + feature) + 0.95) 
                              for feature in range(self.num_features // 2)]
            # print(f"Ranges {ranges_classes} ") # Optional: for debugging
            
            for i in range(samples_per_class):
                sample_index = _class * samples_per_class + i
                # self.labels[sample_index] = _class # This is now done in _generate_labels

                for feat in range(self.num_features):
                    if feat < (self.num_features // 2):
                        # Ensure that the feature index for ranges_classes is within bounds
                        range_idx = feat 
                        range_start, range_finish = ranges_classes[range_idx]
                        structured_data_array[sample_index, feat] = round(np.random.uniform(range_start, range_finish), 2)
                    else:
                        # For the second half of features, generate random uniform data
                        structured_data_array[sample_index, feat] = round(np.random.uniform(100, 100000), 2)
        return structured_data_array

    def _generate_images(self):
        """
        Generate synthetic images with simple patterns and color based on the label.
        Introduces variations in position, color, and size of the main object,
        adds distractor objects, background noise, applies augmentations,
        and can make object visibility dependent on a numerical feature.
        """

        unique_labels = np.unique(self.labels).size
        norm_uniq_labels_colors = [int(np.interp(i, (0, unique_labels), (1, 240))) for i in range(unique_labels)]
        norm_uniq_labels_base_diameter = [int(np.interp(i, (0, unique_labels), (self.image_size[0]//2.5, self.image_size[0]-10))) for i in range(unique_labels)]

        images = []
        for sample_idx, label_value in enumerate(self.labels):
            # Create base image (white background)
            img_array = np.full((self.image_size[1], self.image_size[0], 3), 255, dtype=np.uint8)

            # Add background noise if specified
            if self.background_noise_std > 0:
                noise = np.random.normal(0, self.background_noise_std * 255, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img)

            # --- Determine if the Main Informative Circle should be drawn ---
            draw_main_circle = True # Assume it will be drawn by default
            if self.enable_multimodal_image_relations:
                # Conditional visibility based on a numerical feature
                if self.numerical_features[sample_idx, self.multimodal_feature_idx] < self.multimodal_visibility_threshold:
                    draw_main_circle = False
            
            # --- Main Informative Circle ---
            if draw_main_circle:
                base_color_val = norm_uniq_labels_colors[label_value]
                current_color_val = base_color_val
                if self.vary_circle_color:
                    color_offset = np.random.randint(-self.circle_color_variation_amount, self.circle_color_variation_amount + 1)
                    current_color_val = np.clip(base_color_val + color_offset, 0, 255)
                circle_fill_color = (current_color_val, current_color_val, current_color_val)

                base_diameter = norm_uniq_labels_base_diameter[label_value]
                current_diameter = base_diameter
                if self.vary_circle_size:
                    size_offset_abs = int(self.image_size[0] * self.circle_size_variation_ratio)
                    size_offset = np.random.randint(-size_offset_abs, size_offset_abs + 1)
                    min_diameter = max(5, int(self.image_size[0] * 0.1))
                    max_diameter = self.image_size[0] - 2
                    current_diameter = np.clip(base_diameter + size_offset, min_diameter, max_diameter)
                
                x0, y0 = 10, 10
                if self.vary_circle_position:
                    max_offset_pixels_x = int(self.image_size[0] * self.circle_position_max_offset_ratio)
                    max_offset_pixels_y = int(self.image_size[1] * self.circle_position_max_offset_ratio)
                    offset_x = np.random.randint(-max_offset_pixels_x, max_offset_pixels_x + 1)
                    offset_y = np.random.randint(-max_offset_pixels_y, max_offset_pixels_y + 1)
                    potential_x0 = 10 + offset_x
                    potential_y0 = 10 + offset_y
                    x0 = np.clip(potential_x0, 0, self.image_size[0] - current_diameter -1)
                    y0 = np.clip(potential_y0, 0, self.image_size[1] - current_diameter -1)

                x1 = x0 + current_diameter
                y1 = y0 + current_diameter
                draw.ellipse((x0, y0, x1, y1), fill=circle_fill_color)
            
            # --- Distractor Objects ---
            if self.num_distractor_objects > 0:
                max_distractor_size = int(self.image_size[0] * self.distractor_max_size_ratio)
                min_distractor_size = max(3, int(max_distractor_size * 0.2))
                for _ in range(self.num_distractor_objects):
                    distractor_size = np.random.randint(min_distractor_size, max_distractor_size + 1)
                    distractor_x = np.random.randint(0, self.image_size[0] - distractor_size)
                    distractor_y = np.random.randint(0, self.image_size[1] - distractor_size)
                    distractor_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                    draw.rectangle(
                        [distractor_x, distractor_y, 
                         distractor_x + distractor_size, distractor_y + distractor_size],
                        fill=distractor_color
                    )
            
            # --- Image Augmentations ---
            if self.apply_blur:
                blur_radius = np.random.uniform(self.blur_radius_range[0], self.blur_radius_range[1])
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            if self.apply_brightness_contrast:
                brightness_factor = np.random.uniform(self.brightness_factor_range[0], self.brightness_factor_range[1])
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness_factor)

                contrast_factor = np.random.uniform(self.contrast_factor_range[0], self.contrast_factor_range[1])
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast_factor)

            # The following block was moved up to correctly control circle drawing
            # if self.enable_multimodal_image_relations:
            #     # Conditional visibility based on a numerical feature
            #     # Ensure self.numerical_features is used here
            #     if self.numerical_features[sample_idx, self.multimodal_feature_idx] < self.multimodal_visibility_threshold:
            #         draw_main_circle = False

            images.append(ToTensor()(img).numpy())

        return np.array(images, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.numerical_features[idx], device=device), # Use self.numerical_features
            torch.tensor(self.images[idx], device=device), # Use self.images
            torch.tensor(self.labels[idx], device=device),
        )

