from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pickle as p
import itertools

from scipy import stats

plt.rc('font', family='Nimbus Sans')


def plot_metrics_comparison(method_to_paths: OrderedDict, output_filename: str, fig_title: str,
                            add_legend: bool = False) -> None:
    """
    Plots metrics over time for multiple methods with 95% confidence intervals.
    :param method_to_paths: OrderedDict mapping method names to their result paths.
    :param output_filename: Name of the output file to save the plot.
    :param fig_title: Title for the entire figure.
    :param add_legend: Whether to add a legend to the plot.
    """
    nature_colors = [
        '#2E3440',
        '#85AA69',
        '#5E81AC',
        '#BF616A',
        '#B48EAD',
        '#88C0D0',
        '#D08770',
        '#EBCB8B',
    ]

    method_color_map = {
        'Unconstrained': '#2E3440',
        'Cosine': '#85AA69',
        'RankNet': '#5E81AC',
        'LightGBM': '#BF616A'
    }

    colors = []
    available_colors = nature_colors.copy()
    for method_name in method_to_paths.keys():
        if method_name in method_color_map:
            color = method_color_map[method_name]
            colors.append(color)
            if color in available_colors:
                available_colors.remove(color)
        else:
            color = available_colors.pop(0) if available_colors else nature_colors[len(colors) % len(nature_colors)]
            colors.append(color)
    num_methods = len(method_to_paths)
    metrics_all_methods = []
    method_names = method_to_paths.keys()

    for method, file_path in method_to_paths.items():
        with open(osp.join(file_path, "metrics_over_time_test_mixed.pkl"), "rb") as f:
            total_metrics_over_t_10 = p.load(f)

        mrr_hard = np.array(total_metrics_over_t_10['mrr_hard'])
        hits_at_10 = np.array(total_metrics_over_t_10['hits@10_hard'])
        pairwise_acc = np.array(total_metrics_over_t_10['pairwise_accuracy'])
        ndcg_at_10 = np.array(total_metrics_over_t_10['ndcg@10'])
        ndcg_at_100 = np.array(total_metrics_over_t_10['ndcg@100'])

        metrics_all_methods.append([pairwise_acc, mrr_hard, hits_at_10, ndcg_at_10, ndcg_at_100])

    titles = ['Pairwise Accuracy', 'MRR', 'H@10', 'NDCG@10', 'NDCG@100']
    x_start = [1, 1, 1, 1, 1]
    fig, axes = plt.subplots(1, len(titles),
                             figsize=(10, 2.85 if add_legend else 2.5))  # Increase figure size for better spacing

    for ax, title, metric_idx in zip(axes, titles, range(len(titles))):
        num_timesteps = max([metrics_all_methods[i][metric_idx].shape[1] for i in range(num_methods)])

        if title != "Pairwise Accuracy":
            num_timesteps -= 1
        time_steps = np.arange(x_start[metric_idx], num_timesteps + x_start[metric_idx], 1)

        all_values = []

        for method_idx, (metrics, method_name, color) in enumerate(
                zip(metrics_all_methods, method_names, itertools.cycle(colors))):
            metric = metrics[metric_idx]
            if method_name == "Unconstrained":
                mean_val = metric[:, 0].mean() * 100
                sem = stats.sem(metric[:, 0]) * 100  # Standard error of the mean
                ci = sem * stats.t.ppf((1 + 0.95) / 2., len(metric[:, 0]) - 1)  # 95% CI
                ax.axhline(mean_val, color=color, linestyle='--')
                ax.axhspan(mean_val - ci, mean_val + ci, color=color, alpha=0.2)
                all_values.extend([mean_val - ci, mean_val + ci])
            else:
                if title == "Pairwise Accuracy":
                    mean_vals = metric.mean(axis=0) * 100
                    sem_vals = stats.sem(metric, axis=0) * 100
                    ci_vals = sem_vals * stats.t.ppf((1 + 0.95) / 2., len(metric) - 1)
                    ax.plot(time_steps, mean_vals, marker='.', markersize=10, color=color)
                    ax.fill_between(time_steps, mean_vals - ci_vals, mean_vals + ci_vals,
                                    color=color, alpha=0.2)
                    all_values.extend((mean_vals - ci_vals).tolist())
                    all_values.extend((mean_vals + ci_vals).tolist())
                else:
                    mean_vals = metric.mean(axis=0)[1:] * 100
                    sem_vals = stats.sem(metric, axis=0)[1:] * 100
                    ci_vals = sem_vals * stats.t.ppf((1 + 0.95) / 2., len(metric) - 1)
                    ax.plot(time_steps, mean_vals, marker='.', markersize=10, color=color)
                    ax.fill_between(time_steps, mean_vals - ci_vals, mean_vals + ci_vals,
                                    color=color, alpha=0.2)
                    all_values.extend((mean_vals - ci_vals).tolist())
                    all_values.extend((mean_vals + ci_vals).tolist())
            print(f"{method_name} - Final value of {title}: {metric.mean(axis=0)[-1] * 100:.1f}%")

        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            y_range = max_val - min_val
            y_min = min_val - 0.05 * abs(min_val)
            y_max = max_val + 0.05 * abs(max_val)
            ax.set_ylim(y_min, y_max)

        ax.set_ylabel(f"{title} (%)")
        ax.set_xlabel('t')
        ax.set_xticks(time_steps)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=0.98)
    fig.suptitle(fig_title, y=0.93)
    if add_legend:
        legend_patches = [
            plt.Line2D([0], [0], color=colors[0], lw=2, linestyle='--'),
            *[plt.Line2D([0], [0], color=color, lw=3) for color in colors[1:num_methods]]
        ]

        fig.legend(legend_patches, method_names, loc='lower center', bbox_to_anchor=(0.5, -0.01),
                   ncol=len(method_names))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
    else:
        fig.tight_layout(rect=(0, 0, 1, 1))

    plots_dir = "plots"
    if not osp.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(osp.join(plots_dir, output_filename))
    print(f"Saved plot to {osp.join(plots_dir, output_filename)}")


plot_metrics_comparison(
    OrderedDict([
        ("Unconstrained", "results/fb15k237/unconstrained/test/fb15k237-betae_10_0.0002_default_test_mixed_1760968758_073xj415"),
        ("Cosine", "results/fb15k237/cosine/test/fb15k237-betae_10_0.0002_cosine_0.25_0.5_test_mixed_1761037498_fvwjmdjs"),
        ("LightGBM", "results/fb15k237/lightgbm/test/fb15k237-betae_10_0.0002_lightgbm_lambdamart_test_mixed_1762078248_c2w6wno3"),
        ("RankNet", "results/fb15k237/ranknet/test/fb15k237-betae_10_0.0002_ranknet_0.001_test_mixed_1761053698_5fpk3jiw"),
    ]),
    output_filename="fb15k237.pdf",
    fig_title="FB15k-237",
    add_legend=True
)

# plot_metrics_comparison(
#     OrderedDict([
#         ("Unconstrained", "results/fb15k237-betae_10_0.0002_default_test_mixed_1761210501_yfk5bmu0"),
#         ("Cosine", "results/fb15k237-betae_10_0.0002_cosine_0.25_0.5_test_mixed_1761210367_8hb7ba9w"),
#         ("LambdaRank0", "results/fb15k237-betae_10_0.0002_lightgbm_lambdamart_mixed_1761839940_gh5i43fe"),
#         ("LambdaRankS", "results/fb15k237-betae_10_0.0002_lightgbm_lambdamart_mixed_1761909679_nix87y05"),
#         ("LambdaRank1", "results/fb15k237-betae_10_0.0002_lightgbm_lambdamart_mixed_1761918930_gwbu5tb0"),
#         ("LambdaRank2", "results/fb15k237-betae_10_0.0002_lightgbm_lambdamart_mixed_1761919932_eyo7ddxg")
#     ]),
#     output_filename="fb15k237_1p.pdf",
#     fig_title="FB15k-237",
#     add_legend=True
# )

# plot_metrics_comparison(
#     OrderedDict([
#         ("Unconstrained", "results/hetionet/unconstrained/test/hetionet_10_0.001_default_test_mixed_1760975302_nblg4ymc"),
#         ("Cosine", "results/hetionet/cosine/test/hetionet_10_0.001_cosine_0.25_0.5_test_mixed_1761037524_0fys9umr"),
#         ("RankNet", "results/hetionet/ranknet/test/hetionet_10_0.001_ranknet_0.001_test_mixed_1761056339_3ekqazqe")
#     ]),
#     output_filename="hetionet.pdf",
#     fig_title="Hetionet",
#     add_legend=True
# )
