from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import itertools

plt.rc('font', family='Nimbus Sans')


def plot_metrics_comparison(method_to_paths: OrderedDict, output_filename: str) -> None:
    """
    Plots boxplots of metrics over time for multiple methods.
    :param file_paths: List of file paths containing metrics.
    :param method_names: List of method names corresponding to each file.
    """
    colors = [f"C{i}" for i in range(len(method_to_paths))]  # Define distinct colors
    colors[0] = "black"
    num_methods = len(method_to_paths)
    metrics_all_methods = []
    method_names = method_to_paths.keys()

    for method, file_path in method_to_paths.items():
        with open(file_path, "rb") as f:
            total_metrics_over_t_10 = p.load(f)

        # Extract metrics
        mrr_hard = np.array(total_metrics_over_t_10['mrr_hard'])
        hits_at_10 = np.array(total_metrics_over_t_10['hits@10_hard'])
        pairwise_acc = np.array(total_metrics_over_t_10['pairwise_accuracy'])

        metrics_all_methods.append([pairwise_acc, mrr_hard, hits_at_10])

    # Create subplots for boxplots
    titles = ['Pairwise Accuracy', 'MRR', 'H@10']
    x_start = [1, 0, 0]
    fig, axes = plt.subplots(1, len(titles), figsize=(10, 3.5))  # Increase figure size for better spacing

    # Plot metrics
    for ax, title, metric_idx in zip(axes, titles, range(len(titles))):
        num_timesteps = max([metrics_all_methods[i][metric_idx].shape[1] for i in range(num_methods)])
        time_steps = np.arange(x_start[metric_idx], num_timesteps + x_start[metric_idx], 1)
        for method_idx, (metrics, method_name, color) in enumerate(zip(metrics_all_methods, method_names, itertools.cycle(colors))):
            metric = metrics[metric_idx]
            if method_name == "QTO":
                ax.axhline(metric[:, 0].mean(), color=color, linestyle='--')
            else:
                ax.plot(time_steps, metric.mean(axis=0), marker='.', markersize=10, color=color)

        ax.set_ylabel(title)
        ax.set_xlabel('Number of interactions')
        ax.set_xticks(time_steps)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if title == "Pairwise Accuracy":
            ax.set_ylim([0, 1])

    # Add legend with correct colors
    legend_patches = [
        plt.Line2D([0], [0], color=colors[0], lw=2, linestyle='--'),
        *[plt.Line2D([0], [0], color=color, lw=3 ) for color in colors[1:num_methods]]
    ]

    fig.legend(legend_patches, method_names, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(method_names))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.275)  # Make room for legend
    plt.savefig(output_filename)


plot_metrics_comparison(OrderedDict([
    ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_1746017082/metrics_over_time_test_mixed.pkl"),
    ("Cosine", "results/fb15k237-betae_10_0.0002_cosine_0.1_0.9_test_mixed_1746011235/metrics_over_time_test_mixed.pkl"),
    ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_1746261385_dvbsv30h/metrics_over_time_test_mixed.pkl")
    ]),
    output_filename="fb15k237_over_time.pdf"
)

plot_metrics_comparison(OrderedDict([
    ("QTO", "results/hetionet_10_0.001_default_test_mixed_1746798909_v3m7gqu4/metrics_over_time_test_mixed.pkl"),
    ("Cosine", "results/hetionet_10_0.001_cosine_0.1_0.9_test_mixed_1746799431_5jw6w7yi/metrics_over_time_test_mixed.pkl"),
    ]),
    output_filename="hetionet_over_time.pdf"
)