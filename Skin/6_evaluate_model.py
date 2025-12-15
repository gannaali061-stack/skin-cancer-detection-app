"""
4_evaluate_model.py
Comprehensive model evaluation on test set
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ModelEvaluator:
    def __init__(self, model_path, img_size=224):
        self.img_size = img_size
        self.model = None
        self.model_path = model_path
        self.predictions = None
        self.true_labels = None
        
    def load_model(self):
        """Load trained model"""
        
        print(f"Loading model from: {self.model_path}")
        
        try:
            self.model = keras.models.load_model(self.model_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def create_test_generator(self, batch_size=32):
        """Create test data generator"""
        
        print("\nPreparing test data...")
        
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        test_generator = test_datagen.flow_from_directory(
            'dataset/test',
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"Test samples: {test_generator.samples}")
        print(f"Classes: {test_generator.class_indices}")
        
        return test_generator
    
    def evaluate(self):
        """Evaluate model on test set"""
        
        if self.model is None:
            print("Must load model first!")
            return
        
        print("\n" + "="*60)
        print("EVALUATING MODEL ON TEST SET")
        print("="*60)
        
        test_gen = self.create_test_generator()
        
        print("\nMaking predictions...")
        predictions_prob = self.model.predict(test_gen, verbose=1)
        self.predictions = (predictions_prob > 0.5).astype(int).flatten()
        self.true_labels = test_gen.classes
        
        print("\nCalculating metrics...")
        
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision = precision_score(self.true_labels, self.predictions)
        recall = recall_score(self.true_labels, self.predictions)
        f1 = f1_score(self.true_labels, self.predictions)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print("="*60)
        
        print("\nClassification Report:")
        print(classification_report(
            self.true_labels, 
            self.predictions,
            target_names=['Benign', 'Malignant'],
            digits=4
        ))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': self.predictions,
            'true_labels': self.true_labels,
            'predictions_prob': predictions_prob
        }
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        
        if self.predictions is None:
            print("Must run evaluate() first!")
            return
        
        print("\nPlotting confusion matrix...")
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'],
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        plt.text(
            0.5, -0.15,
            f'TN={cm[0,0]} | FP={cm[0,1]} | FN={cm[1,0]} | TP={cm[1,1]}',
            ha='center',
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("Saved confusion_matrix.png")
        plt.close()
    
    def plot_roc_curve(self, predictions_prob):
        """Plot ROC curve"""
        
        print("\nPlotting ROC curve...")
        
        fpr, tpr, thresholds = roc_curve(self.true_labels, predictions_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
        print("Saved roc_curve.png")
        print(f"AUC Score: {roc_auc:.4f}")
        plt.close()
    
    def plot_prediction_distribution(self, predictions_prob):
        """Plot prediction distribution"""
        
        print("\nPlotting prediction distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].hist(
            predictions_prob[self.true_labels == 0],
            bins=50,
            alpha=0.7,
            label='Benign',
            color='green'
        )
        axes[0].hist(
            predictions_prob[self.true_labels == 1],
            bins=50,
            alpha=0.7,
            label='Malignant',
            color='red'
        )
        axes[0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        axes[0].set_xlabel('Prediction Probability', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        data_to_plot = [
            predictions_prob[self.true_labels == 0].flatten(),
            predictions_prob[self.true_labels == 1].flatten()
        ]
        
        bp = axes[1].boxplot(
            data_to_plot,
            labels=['Benign', 'Malignant'],
            patch_artist=True
        )
        
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        
        axes[1].axhline(y=0.5, color='black', linestyle='--', label='Threshold')
        axes[1].set_ylabel('Prediction Probability', fontsize=12)
        axes[1].set_title('Prediction Probability by Class', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('prediction_distribution.png', dpi=150, bbox_inches='tight')
        print("Saved prediction_distribution.png")
        plt.close()
    
    def save_results(self, results):
        """Save results to files"""
        
        print("\nSaving results...")
        
        metrics_df = pd.DataFrame([{
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1_Score': results['f1_score']
        }])
        
        metrics_df.to_csv('evaluation_metrics.csv', index=False)
        print("Saved evaluation_metrics.csv")
        
        predictions_df = pd.DataFrame({
            'True_Label': results['true_labels'],
            'Predicted_Label': results['predictions'],
            'Prediction_Probability': results['predictions_prob'].flatten()
        })
        
        predictions_df.to_csv('predictions.csv', index=False)
        print("Saved predictions.csv")

def main():
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("ERROR: models folder not found!")
        return
    
    model_files = list(models_dir.glob('*.keras'))
    
    if not model_files:
        print("ERROR: No models found!")
        print("Run: python 3_train_model.py first")
        return
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"\nUsing model: {latest_model.name}")
    
    evaluator = ModelEvaluator(str(latest_model))
    
    if not evaluator.load_model():
        return
    
    results = evaluator.evaluate()
    
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve(results['predictions_prob'])
    evaluator.plot_prediction_distribution(results['predictions_prob'])
    
    evaluator.save_results(results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - prediction_distribution.png")
    print("  - evaluation_metrics.csv")
    print("  - predictions.csv")

if __name__ == "__main__":
    main()