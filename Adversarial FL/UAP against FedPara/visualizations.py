import matplotlib.pyplot as plt
import numpy as np
from utils import CLASSES # Assuming utils.py

plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style

def plot_accuracy_vs_rounds(training_logs, titles, overall_title="Federated Learning Accuracy"):
    """
    Plots accuracy vs. rounds for multiple experimental runs.
    training_logs: A list of training_log dictionaries (or a single log).
    titles: A list of titles for each log in training_logs (or a single title if one log).
    """
    if not isinstance(training_logs, list):
        training_logs = [training_logs]
        titles = [titles]

    plt.figure(figsize=(12, 7))
    
    for i, log in enumerate(training_logs):
        if not log: continue # Skip empty logs
        rounds = [entry['round'] for entry in log]
        global_acc = [entry['global_acc'] for entry in log]
        plt.plot(rounds, global_acc, marker='o', linestyle='-', label=titles[i], linewidth=2, markersize=5)

    plt.xlabel("Communication Rounds", fontsize=14)
    plt.ylabel("Global Accuracy (%)", fontsize=14)
    plt.title(overall_title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_class_accuracy_comparison(class_acc_initial, class_acc_final, partitioning_strategy_name, classes_names=CLASSES):
    """Plots initial vs. final class-wise accuracy."""
    x = np.arange(len(classes_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, class_acc_initial, width, label='Initial (Round 0)', color='skyblue')
    rects2 = ax.bar(x + width/2, class_acc_final, width, label=f'Final (After Attack under {partitioning_strategy_name})', color='salmon')

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xlabel('Classes', fontsize=14)
    ax.set_title(f'Class-wise Accuracy Comparison: {partitioning_strategy_name}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(classes_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.show()


def plot_communication_savings(original_model_size_mb, fedpara_model_size_mb):
    """Plots a bar chart comparing model sizes for communication cost."""
    labels = ['Standard CNN', 'FedPara CNN']
    sizes = [original_model_size_mb, fedpara_model_size_mb]
    
    plt.figure(figsize=(7, 6))
    bars = plt.bar(labels, sizes, color=['cornflowerblue', 'lightcoral'])
    plt.ylabel("Model Size (MB)", fontsize=14)
    plt.title("Communication Cost (Model Size per Round)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(sizes), f'{yval:.2f} MB', ha='center', va='bottom', fontsize=10)
        
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Add other plotting functions if needed (loss trends, attack impact directly)
# The survival_rate function could also go here or in simulation if used for specific metrics.
def calculate_target_class_survival_rate(initial_class_acc, final_class_acc, target_class_idx):
    if initial_class_acc[target_class_idx] == 0:
        return 0.0 if final_class_acc[target_class_idx] == 0 else float('inf') # Or handle as appropriate
    return (final_class_acc[target_class_idx] / initial_class_acc[target_class_idx]) * 100