import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import itertools

plt.rc('font', family='Nimbus Sans')


def plot_metrics_comparison(file_paths, method_names):
    """
    Plots boxplots of metrics over time for multiple methods.
    :param file_paths: List of file paths containing metrics.
    :param method_names: List of method names corresponding to each file.
    """
    assert len(file_paths) == len(method_names), "Number of file paths must match number of method names."

    colors = [f"C{i}" for i in range(len(file_paths))]  # Define distinct colors
    num_methods = len(file_paths)
    metrics_all_methods = []

    for file_path in file_paths:
        with open(file_path, "rb") as f:
            total_metrics_over_t_10 = p.load(f)

        # Extract metrics
        mrr_hard = np.array(total_metrics_over_t_10['mrr_hard'])
        hits_at_10 = np.array(total_metrics_over_t_10['hits@10_hard'])
        pairwise_acc = np.array(total_metrics_over_t_10['pairwise_accuracy'])

        metrics_all_methods.append([pairwise_acc, mrr_hard, hits_at_10])

    # Create subplots for boxplots
    titles = ['Pairwise Accuracy', 'MRR', 'H@10']#, '$\Delta$Hits@10']
    x_start = [1, 0, 0]
    fig, axes = plt.subplots(1, len(titles), figsize=(10, 3.5))  # Increase figure size for better spacing

    # Plot metrics
    for ax, title, metric_idx in zip(axes, titles, range(len(titles))):
        num_timesteps = metrics_all_methods[0][metric_idx].shape[1]
        time_steps = np.arange(x_start[metric_idx], num_timesteps + x_start[metric_idx], 1)
        for method_idx, (metrics, method_name, color) in enumerate(zip(metrics_all_methods, method_names, itertools.cycle(colors))):
            metric = metrics[metric_idx]
            ax.plot(time_steps, metric.mean(axis=0), marker='.', markersize=10)


        ax.set_title(title)
        ax.set_xlabel('Number of interactions')
        ax.set_xticks(time_steps)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add legend with correct colors
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4) for color in colors[:num_methods]]
    fig.legend(legend_patches, method_names, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(method_names))

    fig.suptitle('Distribution of Metrics Over Time Across Methods', y=1.05)
    plt.tight_layout()
    plt.show()


plot_metrics_comparison([
    "results/fb15k237-betae_10_0.0002_greedy_test_mixed_1745998797/metrics_over_time_test_mixed.pkl",
    "results/fb15k237-betae_10_0.0002_cosine_0.1_0.9_test_mixed_1746011235/metrics_over_time_test_mixed.pkl",
    "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_1746020972/metrics_over_time_test_mixed.pkl"
],
    [
        "Greedy",
        "Cosine",
        "NQR"
])
