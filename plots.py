from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
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
        with open(osp.join(file_path, "metrics_over_time_test_mixed.pkl"), "rb") as f:
            total_metrics_over_t_10 = p.load(f)

        # Extract metrics
        mrr_hard = np.array(total_metrics_over_t_10['mrr_hard'])
        hits_at_10 = np.array(total_metrics_over_t_10['hits@10_hard'])
        pairwise_acc = np.array(total_metrics_over_t_10['pairwise_accuracy'])
        ndcg_at_10 = np.array(total_metrics_over_t_10['ndcg@10'])
        ndcg_at_100 = np.array(total_metrics_over_t_10['ndcg@100'])

        metrics_all_methods.append([pairwise_acc, mrr_hard, hits_at_10, ndcg_at_10, ndcg_at_100])

    # Create subplots for boxplots
    titles = ['Pairwise Accuracy', 'MRR', 'H@10', 'NDCG@10']
    x_start = [1, 1, 1, 1]
    fig, axes = plt.subplots(1, len(titles), figsize=(10, 3.5))  # Increase figure size for better spacing

    # Plot metrics
    for ax, title, metric_idx in zip(axes, titles, range(len(titles))):
        num_timesteps = max([metrics_all_methods[i][metric_idx].shape[1] for i in range(num_methods)])

        if title != "Pairwise Accuracy":
            num_timesteps -= 1
        time_steps = np.arange(x_start[metric_idx], num_timesteps + x_start[metric_idx], 1)
        for method_idx, (metrics, method_name, color) in enumerate(zip(metrics_all_methods, method_names, itertools.cycle(colors))):
            metric = metrics[metric_idx]
            if method_name == "QTO":
                ax.axhline(metric[:, 0].mean(), color=color, linestyle='--')
            else:
                if title == "Pairwise Accuracy":
                    ax.plot(time_steps, metric.mean(axis=0), marker='.', markersize=10, color=color)
                else:
                    ax.plot(time_steps, metric.mean(axis=0)[1:], marker='.', markersize=10, color=color)
            print(f"{method_name} - Final value of {title}: {metric.mean(axis=0)[-1]}")

        ax.set_ylabel(title)
        ax.set_xlabel('Number of interactions')
        ax.set_xticks(time_steps)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # if title == "Pairwise Accuracy":
        ax.set_ylim([0, 1])

    # Add legend with correct colors
    legend_patches = [
        plt.Line2D([0], [0], color=colors[0], lw=2, linestyle='--'),
        *[plt.Line2D([0], [0], color=color, lw=3 ) for color in colors[1:num_methods]]
    ]

    fig.legend(legend_patches, method_names , loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(method_names))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.275)  # Make room for legend
    plt.savefig(output_filename)


# TARGET MODE PLOTS (4 methods: QTO, MeanCosine, NQR, SCORE)

# FB15K237 - Target Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_target_1757178216_rq8chcd6"),
    ("MeanCosine", "results/fb15k237-betae_10_0.0002_cosine_mean_test_mixed_target_1757240412_4fkuo5mv"),
    ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_target_1757332560_2s3z0uom"),
    ("SCORE", "results/fb15k237-betae_10_0.0002_score_test_mixed_target_1757349746_vcvn45r2")
]),
output_filename="fb15k237_target_comparison.pdf"
)

# HETIONET - Target Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/hetionet_10_0.001_default_test_mixed_target_1757178246_dxkau1zu"),
    ("MeanCosine", "results/hetionet_10_0.001_cosine_mean_test_mixed_target_1757240403_6n5n8deo"),
    ("NQR", "results/hetionet_10_0.001_nqr_0.001_test_mixed_target_1757350038_cd4q5jcy"),
    ("SCORE", "results/hetionet_10_0.001_score_test_mixed_target_1757349853_6opdbv9s")
]),
output_filename="hetionet_target_comparison.pdf"
)

# NELL995 - Target Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/nell-betae_10_0.0002_default_test_mixed_target_1757178211_0pf4oo3l"),
    ("MeanCosine", "results/nell-betae_10_0.0002_cosine_mean_test_mixed_target_1757240402_p0iy5bvo"),
    ("NQR", "results/nell-betae_10_0.0002_nqr_0.001_test_mixed_target_1757350066_70pd7fnx"),
    ("SCORE", "results/nell-betae_10_0.0002_score_test_mixed_target_1757349914_mim7r1ox")
]),
output_filename="nell995_target_comparison.pdf"
)

# FULL MODE PLOTS (2 methods: QTO, SCORE)

# FB15K237 - Full Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_full_1757178217_q1f9v156"),
    ("SCORE", "results/fb15k237-betae_10_0.0002_score_test_mixed_full_1757349830_k9k9hl3w")
]),
output_filename="fb15k237_full_comparison.pdf"
)

# HETIONET - Full Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/hetionet_10_0.001_default_test_mixed_full_1757178247_2vbjrw1k"),
    ("SCORE", "results/hetionet_10_0.001_score_test_mixed_full_1757349853_s6l28qfp")
]),
output_filename="hetionet_full_comparison.pdf"
)

# NELL995 - Full Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/nell-betae_10_0.0002_default_test_mixed_full_1757178218_9jqnj61o"),
    ("SCORE", "results/nell-betae_10_0.0002_score_test_mixed_full_1757349943_q3v0uj0e")
]),
output_filename="nell995_full_comparison.pdf"
)