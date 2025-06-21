import numpy as np
import cv2

# Dicionário de formas por classe
SHAPES_BY_CLASS = {
    0: "star",
    1: "square",
    2: "circle",
    3: "triangle",
    4: "diamond",
    5: "cross",
    6: "pentagon"
}

def draw_shape(img, shape, center, size, color):
    # Garante que color é uma tupla de int
    color = tuple(int(c) for c in color)
    if shape == "circle":
        cv2.circle(img, tuple(map(int, center)), int(size), color, -1)
    elif shape == "square":
        top_left = (int(center[0] - size), int(center[1] - size))
        bottom_right = (int(center[0] + size), int(center[1] + size))
        cv2.rectangle(img, top_left, bottom_right, color, -1)
    elif shape == "star":
        pts = []
        for i in range(10):
            angle = i * np.pi / 5
            r = size if i % 2 == 0 else size // 2
            x = int(center[0] + r * np.cos(angle - np.pi/2))
            y = int(center[1] + r * np.sin(angle - np.pi/2))
            pts.append((x, y))
        cv2.fillPoly(img, [np.array(pts, np.int32)], color)
    elif shape == "triangle":
        pts = np.array([
            (int(center[0]), int(center[1] - size)),
            (int(center[0] - size), int(center[1] + size)),
            (int(center[0] + size), int(center[1] + size))
        ], np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == "diamond":
        pts = np.array([
            (int(center[0]), int(center[1] - size)),
            (int(center[0] - size), int(center[1])),
            (int(center[0]), int(center[1] + size)),
            (int(center[0] + size), int(center[1]))
        ], np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == "cross":
        thickness = max(1, int(size // 3))
        cv2.line(img, (int(center[0] - size), int(center[1])), (int(center[0] + size), int(center[1])), color, thickness)
        cv2.line(img, (int(center[0]), int(center[1] - size)), (int(center[0]), int(center[1] + size)), color, thickness)
    elif shape == "pentagon":
        pts = []
        for i in range(5):
            angle = 2 * np.pi * i / 5 - np.pi / 2
            x = int(center[0] + size * np.cos(angle))
            y = int(center[1] + size * np.sin(angle))
            pts.append((x, y))
        cv2.fillPoly(img, [np.array(pts, np.int32)], color)

def generate_synthetic_image(class_idx, img_size=64):
    assert 0 <= class_idx < 7, "Classe deve estar entre 0 e 6"
    shape = SHAPES_BY_CLASS[class_idx]
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # Sorteia tamanho e posição aleatórios
    size = np.random.randint(img_size // 8, img_size // 3)
    margin = size + 2
    center = (
        np.random.randint(margin, img_size - margin),
        np.random.randint(margin, img_size - margin)
    )
    color = (255, 255, 255)  # Branco
    draw_shape(img, shape, center, size, color)
    return img