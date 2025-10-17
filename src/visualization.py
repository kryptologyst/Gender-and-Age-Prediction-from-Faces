"""
Explainability and visualization utilities for gender and age prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """Visualization utilities for prediction results."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = {
            'male': '#3498db',
            'female': '#e74c3c',
            'face_box': '#2ecc71',
            'text': '#2c3e50'
        }
    
    def visualize_prediction(self, 
                           image: np.ndarray, 
                           result: Dict, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive visualization of prediction results.
        
        Args:
            image: Original input image
            result: Prediction result dictionary
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gender & Age Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Original image with face detection
        ax1 = axes[0, 0]
        display_image = image.copy()
        
        if result.get('faces'):
            for i, (x, y, w, h) in enumerate(result['faces']):
                cv2.rectangle(display_image, (x, y), (x + w, y + h), 
                            (0, 255, 0), 2)
                label = f"{result['gender']}, {result['age']}"
                cv2.putText(display_image, label, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ax1.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Face Detection & Prediction')
        ax1.axis('off')
        
        # Gender confidence
        ax2 = axes[0, 1]
        if 'gender_confidence' in result:
            gender_labels = ['Male', 'Female']
            gender_conf = result['gender_confidence']
            other_conf = 1 - gender_conf
            
            bars = ax2.bar(gender_labels, [gender_conf, other_conf], 
                          color=[self.colors['male'], self.colors['female']], 
                          alpha=0.7)
            ax2.set_title(f'Gender Confidence: {gender_conf:.2%}')
            ax2.set_ylabel('Confidence')
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, conf in zip(bars, [gender_conf, other_conf]):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{conf:.2%}', ha='center', va='bottom')
        
        # Age distribution
        ax3 = axes[1, 0]
        if 'age_confidence' in result:
            age_buckets = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
            age_conf = result['age_confidence']
            
            # Create a simple distribution visualization
            bucket_index = age_buckets.index(result['age']) if result['age'] in age_buckets else 0
            age_dist = np.zeros(len(age_buckets))
            age_dist[bucket_index] = age_conf
            
            bars = ax3.bar(range(len(age_buckets)), age_dist, 
                          color='skyblue', alpha=0.7)
            ax3.set_title(f'Age Range: {result["age"]} ({age_conf:.2%})')
            ax3.set_ylabel('Confidence')
            ax3.set_xlabel('Age Buckets')
            ax3.set_xticks(range(len(age_buckets)))
            ax3.set_xticklabels(age_buckets, rotation=45)
            ax3.set_ylim(0, 1)
            
            # Highlight the predicted age bucket
            bars[bucket_index].set_color('orange')
        
        # Overall confidence and metrics
        ax4 = axes[1, 1]
        metrics = []
        values = []
        
        if 'overall_confidence' in result:
            metrics.append('Overall\nConfidence')
            values.append(result['overall_confidence'])
        
        if 'faces' in result:
            metrics.append('Faces\nDetected')
            values.append(len(result['faces']))
        
        if values:
            bars = ax4.bar(metrics, values, color='lightcoral', alpha=0.7)
            ax4.set_title('Prediction Metrics')
            ax4.set_ylabel('Value')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}' if isinstance(value, float) else f'{value}',
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        return fig
    
    def create_confidence_heatmap(self, 
                                results: List[Dict], 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap of confidence scores across multiple predictions.
        
        Args:
            results: List of prediction results
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if not results:
            logger.warning("No results provided for heatmap")
            return None
        
        # Extract confidence scores
        gender_confs = []
        age_confs = []
        overall_confs = []
        
        for result in results:
            gender_confs.append(result.get('gender_confidence', 0))
            age_confs.append(result.get('age_confidence', 0))
            overall_confs.append(result.get('overall_confidence', 0))
        
        # Create heatmap data
        data = np.array([gender_confs, age_confs, overall_confs])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([f"Image {i+1}" for i in range(len(results))])
        ax.set_yticks(range(3))
        ax.set_yticklabels(['Gender Confidence', 'Age Confidence', 'Overall Confidence'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Confidence Score')
        
        # Add text annotations
        for i in range(3):
            for j in range(len(results)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Confidence Scores Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to: {save_path}")
        
        return fig


class ModelExplainer:
    """Explainability utilities for understanding model predictions."""
    
    def __init__(self, model, device):
        """
        Initialize model explainer.
        
        Args:
            model: PyTorch model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate_gradcam(self, 
                        image_tensor: torch.Tensor, 
                        target_layer: str = "backbone") -> np.ndarray:
        """
        Generate Grad-CAM visualization for model explainability.
        
        Args:
            image_tensor: Input image tensor
            target_layer: Name of the target layer for Grad-CAM
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        # This is a simplified Grad-CAM implementation
        # In practice, you'd use libraries like captum for more sophisticated methods
        
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_()
        
        # Forward pass
        gender_logits, age_logits = self.model(image_tensor)
        
        # Get the target class (highest confidence)
        target_class = gender_logits.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        gender_logits[0, target_class].backward(retain_graph=True)
        
        # Get gradients
        gradients = image_tensor.grad.data
        
        # Generate heatmap (simplified)
        heatmap = torch.mean(torch.abs(gradients), dim=1).squeeze()
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), 
                              size=(224, 224), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().numpy()
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
    
    def visualize_attention(self, 
                           image: np.ndarray, 
                           heatmap: np.ndarray, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize attention/importance map overlaid on the original image.
        
        Args:
            image: Original input image
            heatmap: Attention/importance heatmap
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet', alpha=0.8)
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        overlay = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to: {save_path}")
        
        return fig


def create_prediction_report(results: List[Dict], 
                           output_dir: Union[str, Path]) -> str:
    """
    Create a comprehensive prediction report with visualizations.
    
    Args:
        results: List of prediction results
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = PredictionVisualizer()
    
    # Create individual visualizations
    for i, result in enumerate(results):
        if 'image_path' in result and Path(result['image_path']).exists():
            image = cv2.imread(result['image_path'])
            if image is not None:
                fig = visualizer.visualize_prediction(image, result)
                save_path = output_dir / f"prediction_{i:03d}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    # Create summary heatmap
    if len(results) > 1:
        fig = visualizer.create_confidence_heatmap(results)
        if fig:
            save_path = output_dir / "confidence_heatmap.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # Create text report
    report_path = output_dir / "prediction_report.txt"
    with open(report_path, 'w') as f:
        f.write("Gender & Age Prediction Report\n")
        f.write("=" * 40 + "\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Image {i+1}:\n")
            f.write(f"  Gender: {result.get('gender', 'Unknown')}\n")
            f.write(f"  Age: {result.get('age', 'Unknown')}\n")
            f.write(f"  Confidence: {result.get('overall_confidence', 0):.2%}\n")
            f.write(f"  Faces Detected: {len(result.get('faces', []))}\n")
            f.write("\n")
    
    logger.info(f"Prediction report created in: {output_dir}")
    return str(report_path)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    sample_result = {
        'gender': 'Male',
        'age': '25-32',
        'gender_confidence': 0.85,
        'age_confidence': 0.78,
        'overall_confidence': 0.815,
        'faces': [(50, 50, 100, 100)]
    }
    
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Create visualization
    visualizer = PredictionVisualizer()
    fig = visualizer.visualize_prediction(sample_image, sample_result)
    plt.show()
