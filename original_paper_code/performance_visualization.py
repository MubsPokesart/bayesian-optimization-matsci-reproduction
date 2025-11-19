# Performance Visualization.ipynb

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Sample data generation for demonstration
np.random.seed(42)
n_samples = 1000

# Generate sample predictions and true labels
y_true = np.random.randint(0, 2, n_samples)
y_pred_proba = np.random.rand(n_samples)
y_pred = (y_pred_proba > 0.5).astype(int)

# 1. Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive']):
    """Plot confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_true, y_pred)

# 2. ROC Curve
def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve with AUC score"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_roc_curve(y_true, y_pred_proba)

# 3. Training History Visualization
def plot_training_history(history_dict):
    """Plot training and validation metrics over epochs"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history_dict['train_loss'], label='Training Loss', marker='o')
    axes[0].plot(history_dict['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history_dict['train_acc'], label='Training Accuracy', marker='o')
    axes[1].plot(history_dict['val_acc'], label='Validation Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Sample training history
history = {
    'train_loss': [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25],
    'val_loss': [0.75, 0.58, 0.52, 0.48, 0.45, 0.44, 0.43, 0.42],
    'train_acc': [0.6, 0.7, 0.75, 0.8, 0.83, 0.85, 0.87, 0.89],
    'val_acc': [0.62, 0.68, 0.72, 0.76, 0.78, 0.79, 0.80, 0.81]
}

plot_training_history(history)

# 4. Performance Metrics Comparison
def plot_metrics_comparison(models_metrics):
    """Compare multiple models' performance metrics"""
    df = pd.DataFrame(models_metrics)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.2
    
    ax.bar(x - width*1.5, df['accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x - width/2, df['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + width/2, df['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width*1.5, df['f1_score'], width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'])
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# Sample metrics for different models
models_metrics = {
    'model': ['Logistic Regression', 'Random Forest', 'SVM', 'Neural Network'],
    'accuracy': [0.82, 0.87, 0.85, 0.89],
    'precision': [0.80, 0.85, 0.83, 0.88],
    'recall': [0.78, 0.84, 0.82, 0.87],
    'f1_score': [0.79, 0.845, 0.825, 0.875]
}

plot_metrics_comparison(models_metrics)

print("Performance visualizations completed successfully!")