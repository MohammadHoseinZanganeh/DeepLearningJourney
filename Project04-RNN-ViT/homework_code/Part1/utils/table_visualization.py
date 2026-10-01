import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
from sklearn.metrics import classification_report
import io

def save_report_as_image(report_dict, model_name, save_path="results/"):
    """
    Convert classification report to a table image
    
    Args:
        report_dict: Classification report as dictionary
        model_name: Name of the model
        save_path: Directory to save the image
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare data for table
    labels = []
    precision = []
    recall = []
    f1 = []
    support = []
    
    for label, metrics in report_dict.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            labels.append(label)
            precision.append(f"{metrics['precision']:.2f}")
            recall.append(f"{metrics['recall']:.2f}")
            f1.append(f"{metrics['f1-score']:.2f}")
            support.append(str(metrics['support']))
    
    # Add average rows
    if 'macro avg' in report_dict:
        labels.append('macro avg')
        precision.append(f"{report_dict['macro avg']['precision']:.2f}")
        recall.append(f"{report_dict['macro avg']['recall']:.2f}")
        f1.append(f"{report_dict['macro avg']['f1-score']:.2f}")
        support.append(str(report_dict['macro avg']['support']))
    
    if 'weighted avg' in report_dict:
        labels.append('weighted avg')
        precision.append(f"{report_dict['weighted avg']['precision']:.2f}")
        recall.append(f"{report_dict['weighted avg']['recall']:.2f}")
        f1.append(f"{report_dict['weighted avg']['f1-score']:.2f}")
        support.append(str(report_dict['weighted avg']['support']))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, len(labels) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    for i in range(len(labels)):
        table_data.append([labels[i], precision[i], recall[i], f1[i], support[i]])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Label', 'Precision', 'Recall', 'F1-Score', 'Support'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.10]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(labels) + 1):
        if i % 2 == 0:
            for j in range(5):
                table[(i, j)].set_facecolor('#D6E4F0')
        else:
            for j in range(5):
                table[(i, j)].set_facecolor('#ECF3F9')
    
    # Color average rows
    for i in range(1, len(labels) + 1):
        if labels[i-1] in ['macro avg', 'weighted avg']:
            for j in range(5):
                table[(i, j)].set_facecolor('#FFC000')
                table[(i, j)].set_text_props(weight='bold')
    
    # Add title
    plt.title(f'Classification Report - {model_name.upper()}', 
              fontsize=14, weight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as image
    filename = f"{save_path}/classification_report_{model_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Classification report saved as image: {filename}")
    return filename


def save_report_as_text(report_str, model_name, save_path="results/"):
    """
    Save classification report as text file
    
    Args:
        report_str: Classification report as string
        model_name: Name of the model
        save_path: Directory to save the text file
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    filename = f"{save_path}/classification_report_{model_name}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Classification Report - {model_name.upper()}\n")
        f.write("="*60 + "\n")
        f.write(report_str)
    
    print(f" Classification report saved as text: {filename}")
    return filename