import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm
from typing import Tuple


class GradCAM:
    def __init__(self, model, layer_name: str = None):
        """
        Initialize Grad-CAM with a model and target layer.

        Args:
            model: Keras model
            layer_name: Name of the convolutional layer to visualize.
                       If None, uses the last convolutional layer.
        """
        self.model = model

        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break

        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )

    def generate_heatmap(self, img_array: np.ndarray, pred_index: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            img_array: Input image array (1, 224, 224, 3) normalized to [0, 1]
            pred_index: Index of target class (None for binary classification)

        Returns:
            Heatmap of shape (224, 224)
        """
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, img_array: np.ndarray, heatmap: np.ndarray,
                       alpha: float = 0.4) -> Image.Image:
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            img_array: Original preprocessed image (1, 224, 224, 3)
            heatmap: Grad-CAM heatmap
            alpha: Blending factor (0-1)

        Returns:
            PIL Image with overlay
        """
        if img_array.shape[0] == 1:
            img_array = img_array[0]

        img_uint8 = (img_array * 255).astype(np.uint8)

        heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)

        return Image.fromarray(overlay)


def generate_gradcam_visualization(img_array: np.ndarray, model,
                                   pred_score: float) -> Tuple[Image.Image, str]:
    """
    Generate Grad-CAM visualization with interpretation.

    Args:
        img_array: Preprocessed image (1, 224, 224, 3)
        model: Trained Keras model
        pred_score: Model prediction score

    Returns:
        Tuple of (PIL Image with overlay, interpretation text)
    """
    try:
        grad_cam = GradCAM(model)

        heatmap = grad_cam.generate_heatmap(img_array)
        overlay_image = grad_cam.overlay_heatmap(img_array, heatmap, alpha=0.5)

        heatmap_intensity = np.mean(heatmap)

        if pred_score > 0.5:
            diagnosis = "Pneumonia"
            if heatmap_intensity > 0.4:
                interpretation = "Strong localized regions of concern detected. AI focused on specific lung areas showing pneumonia indicators."
            else:
                interpretation = "Diffuse pattern detected across lung regions consistent with pneumonia."
        else:
            diagnosis = "Normal"
            if heatmap_intensity > 0.3:
                interpretation = "Areas of normal lung tissue highlighted. Low activation indicates absence of pneumonia markers."
            else:
                interpretation = "Uniform lung patterns detected. Consistent with healthy chest X-ray."

        return overlay_image, interpretation

    except Exception as e:
        raise Exception(f"Grad-CAM generation failed: {str(e)}")


def create_comparison_image(original: np.ndarray, gradcam_overlay: Image.Image) -> Image.Image:
    """
    Create side-by-side comparison of original and Grad-CAM overlay.

    Args:
        original: Original preprocessed image array
        gradcam_overlay: Grad-CAM overlay PIL Image

    Returns:
        Combined comparison PIL Image
    """
    if original.shape[0] == 1:
        original = original[0]

    original_uint8 = (original * 255).astype(np.uint8)
    original_pil = Image.fromarray(original_uint8)

    w, h = original_pil.size
    comparison = Image.new('RGB', (w * 2, h))
    comparison.paste(original_pil, (0, 0))
    comparison.paste(gradcam_overlay, (w, 0))

    return comparison
