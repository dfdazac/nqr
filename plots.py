from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pickle as p
import itertools
from scipy import stats

plt.rc('font', family='Nimbus Sans')


def plot_metrics_comparison(method_to_paths: OrderedDict, output_filename: str, fig_title: str, add_legend: bool = False) -> None:
    """
    Plots metrics over time for multiple methods with 95% confidence intervals.
    :param method_to_paths: OrderedDict mapping method names to their result paths.
    :param output_filename: Name of the output file to save the plot.
    :param fig_title: Title for the entire figure.
    :param add_legend: Whether to add a legend to the plot.
    """
    # Professional color palette suitable for Nature papers
    # These colors are colorblind-friendly, distinguishable, and aesthetically pleasing
    nature_colors = [
        '#2E3440',  # Dark slate gray (professional black alternative)
        '#85AA69',  # Sage green
        '#5E81AC',  # Steel blue
        '#BF616A',  # Muted red
        '#B48EAD',  # Muted purple
        '#88C0D0',  # Light blue
        '#D08770',  # Muted orange
        '#EBCB8B',  # Warm yellow
    ]
    
    # Method-specific color mapping to ensure SCORE always gets muted red
    method_color_map = {
        'SCORE': '#BF616A',  # Muted red - always for SCORE
        'QTO': '#2E3440',    # Dark slate gray
        'MeanCosine': '#85AA69',  # Sage green
        'NQR': '#5E81AC',    # Steel blue
    }
    
    # Assign colors based on method names, falling back to sequential assignment
    colors = []
    available_colors = nature_colors.copy()
    for method_name in method_to_paths.keys():
        if method_name in method_color_map:
            color = method_color_map[method_name]
            colors.append(color)
            # Remove the assigned color from available colors to avoid duplicates
            if color in available_colors:
                available_colors.remove(color)
        else:
            # Assign next available color for unknown methods
            color = available_colors.pop(0) if available_colors else nature_colors[len(colors) % len(nature_colors)]
            colors.append(color)
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
    fig, axes = plt.subplots(1, len(titles), figsize=(10, 2.85 if add_legend else 2.5))  # Increase figure size for better spacing

    # Plot metrics
    for ax, title, metric_idx in zip(axes, titles, range(len(titles))):
        num_timesteps = max([metrics_all_methods[i][metric_idx].shape[1] for i in range(num_methods)])

        if title != "Pairwise Accuracy":
            num_timesteps -= 1
        time_steps = np.arange(x_start[metric_idx], num_timesteps + x_start[metric_idx], 1)
        
        # Collect all values for this metric to determine y-axis limits
        all_values = []
        
        for method_idx, (metrics, method_name, color) in enumerate(zip(metrics_all_methods, method_names, itertools.cycle(colors))):
            metric = metrics[metric_idx]
            if method_name == "QTO":
                # For QTO, show horizontal line with confidence interval as a shaded band
                mean_val = metric[:, 0].mean() * 100
                sem = stats.sem(metric[:, 0]) * 100  # Standard error of the mean
                ci = sem * stats.t.ppf((1 + 0.95) / 2., len(metric[:, 0]) - 1)  # 95% CI
                ax.axhline(mean_val, color=color, linestyle='--')
                ax.axhspan(mean_val - ci, mean_val + ci, color=color, alpha=0.2)
                # Collect values including confidence interval bounds
                all_values.extend([mean_val - ci, mean_val + ci])
            else:
                if title == "Pairwise Accuracy":
                    mean_vals = metric.mean(axis=0) * 100
                    sem_vals = stats.sem(metric, axis=0) * 100
                    ci_vals = sem_vals * stats.t.ppf((1 + 0.95) / 2., len(metric) - 1)
                    ax.plot(time_steps, mean_vals, marker='.', markersize=10, color=color)
                    ax.fill_between(time_steps, mean_vals - ci_vals, mean_vals + ci_vals, 
                                  color=color, alpha=0.2)
                    # Collect values including confidence interval bounds
                    all_values.extend((mean_vals - ci_vals).tolist())
                    all_values.extend((mean_vals + ci_vals).tolist())
                else:
                    mean_vals = metric.mean(axis=0)[1:] * 100
                    sem_vals = stats.sem(metric, axis=0)[1:] * 100
                    ci_vals = sem_vals * stats.t.ppf((1 + 0.95) / 2., len(metric) - 1)
                    ax.plot(time_steps, mean_vals, marker='.', markersize=10, color=color)
                    ax.fill_between(time_steps, mean_vals - ci_vals, mean_vals + ci_vals, 
                                  color=color, alpha=0.2)
                    # Collect values including confidence interval bounds
                    all_values.extend((mean_vals - ci_vals).tolist())
                    all_values.extend((mean_vals + ci_vals).tolist())
            print(f"{method_name} - Final value of {title}: {metric.mean(axis=0)[-1] * 100:.1f}%")

        # Set y-axis limits based on collected values
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            y_range = max_val - min_val
            y_min = min_val - 0.05 * abs(min_val)
            y_max = max_val + 0.05 * abs(max_val)
            ax.set_ylim(y_min, y_max)

        ax.set_ylabel(f"{title} (%)")
        ax.set_xlabel('Number of interactions')
        ax.set_xticks(time_steps)
        
        # Format y-axis ticks to show one decimal place
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add title
    # fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=0.98)
    fig.suptitle(fig_title, y=0.93)
    if add_legend:
        legend_patches = [
            plt.Line2D([0], [0], color=colors[0], lw=2, linestyle='--'),
            *[plt.Line2D([0], [0], color=color, lw=3 ) for color in colors[1:num_methods]]
        ]

        fig.legend(legend_patches, method_names , loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=len(method_names))
        fig.tight_layout(rect=(0, 0.1, 1, 1))  # leave space at bottom and top
    else:
        fig.tight_layout(rect=(0, 0, 1, 1))  # leave space at top for title
    plt.savefig(output_filename)


# TARGET MODE PLOTS (4 methods: QTO, MeanCosine, NQR, SCORE)

# FB15K237 - Target Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_target_1757178216_rq8chcd6"),
    ("MeanCosine", "results/fb15k237-betae_10_0.0002_cosine_mean_test_mixed_target_1757240412_4fkuo5mv"),
    ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_target_1757332560_2s3z0uom"),
    ("SCORE", "results/fb15k237-betae_10_0.0002_score_test_mixed_target_1757349746_vcvn45r2")
]),
output_filename="fb15k237_target_comparison.pdf",
fig_title="FB15K-237"
)

# HETIONET - Target Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/hetionet_10_0.001_default_test_mixed_target_1757178246_dxkau1zu"),
    ("MeanCosine", "results/hetionet_10_0.001_cosine_mean_test_mixed_target_1757240403_6n5n8deo"),
    ("NQR", "results/hetionet_10_0.001_nqr_0.001_test_mixed_target_1757350038_cd4q5jcy"),
    ("SCORE", "results/hetionet_10_0.001_score_test_mixed_target_1757349853_6opdbv9s")
]),
output_filename="hetionet_target_comparison.pdf",
fig_title="Hetionet"
)

# NELL995 - Target Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/nell-betae_10_0.0002_default_test_mixed_target_1757178211_0pf4oo3l"),
    ("MeanCosine", "results/nell-betae_10_0.0002_cosine_mean_test_mixed_target_1757240402_p0iy5bvo"),
    ("NQR", "results/nell-betae_10_0.0002_nqr_0.001_test_mixed_target_1757350066_70pd7fnx"),
    ("SCORE", "results/nell-betae_10_0.0002_score_test_mixed_target_1757349914_mim7r1ox")
]),
output_filename="nell995_target_comparison.pdf", add_legend=True,
fig_title="NELL-995"
)

# FULL MODE PLOTS (2 methods: QTO, SCORE)

# FB15K237 - Full Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_full_1757178217_q1f9v156"),
    ("SCORE", "results/fb15k237-betae_10_0.0002_score_test_mixed_full_1757349830_k9k9hl3w")
]),
output_filename="fb15k237_full_comparison.pdf",
fig_title="FB15K-237"
)

# HETIONET - Full Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/hetionet_10_0.001_default_test_mixed_full_1757178247_2vbjrw1k"),
    ("SCORE", "results/hetionet_10_0.001_score_test_mixed_full_1757349853_s6l28qfp")
]),
output_filename="hetionet_full_comparison.pdf",
fig_title="Hetionet"
)

# NELL995 - Full Mode
plot_metrics_comparison(OrderedDict([
    ("QTO", "results/nell-betae_10_0.0002_default_test_mixed_full_1757178218_9jqnj61o"),
    ("SCORE", "results/nell-betae_10_0.0002_score_test_mixed_full_1757349943_q3v0uj0e")
]),
output_filename="nell995_full_comparison.pdf", add_legend=True,
fig_title="NELL-995"
)