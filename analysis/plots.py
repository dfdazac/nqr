from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import os
import os.path as osp
import pickle as p
import itertools

from scipy import stats

plt.rc('font', family='Nimbus Sans')


def plot_metrics_comparison(method_to_paths: OrderedDict, output_filename: str, fig_title: str,
                            add_legend: bool = False, split: str = 'test') -> None:
    """
    Plots metrics over time for multiple methods with 95% confidence intervals.
    :param method_to_paths: OrderedDict mapping method names to their result paths.
    :param output_filename: Name of the output file to save the plot.
    :param fig_title: Title for the entire figure.
    :param add_legend: Whether to add a legend to the plot.
    :param split: Dataset split to use ('test' or 'valid').
    """
    base_colors = [
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
        'Cosine': '#5E81AC',
        'RankNet': '#BF616A',
        'LightGBM': '#85AA69'
    }

    colors = []
    available_colors = base_colors.copy()
    for method_name in method_to_paths.keys():
        if method_name in method_color_map:
            color = method_color_map[method_name]
            colors.append(color)
            if color in available_colors:
                available_colors.remove(color)
        else:
            color = available_colors.pop(0) if available_colors else base_colors[len(colors) % len(base_colors)]
            colors.append(color)
    num_methods = len(method_to_paths)
    metrics_all_methods = []
    method_names = method_to_paths.keys()

    for method, file_path in method_to_paths.items():
        with open(osp.join(file_path, f"metrics_over_time_{split}_mixed.pkl"), "rb") as f:
            total_metrics_over_t_10 = p.load(f)

        mrr_hard = np.array(total_metrics_over_t_10['mrr_hard'])
        hits_at_10 = np.array(total_metrics_over_t_10['hits@10_hard'])
        pairwise_acc = np.array(total_metrics_over_t_10['pairwise_accuracy'])
        ndcg_at_10 = np.array(total_metrics_over_t_10['ndcg@10'])
        ndcg_at_100 = np.array(total_metrics_over_t_10['ndcg@100'])

        metrics_all_methods.append([pairwise_acc, mrr_hard, hits_at_10, ndcg_at_10, ndcg_at_100])

    titles = ['Pairwise Accuracy', 'MRR', 'NDCG@10']
    titles = ['Pairwise Accuracy', 'MRR', 'NDCG@10']
    x_start = [1, 1, 1, 1]
    fig, axes = plt.subplots(1, len(titles),
                             figsize=(8, 2.5 if add_legend else 2.0))  # Increase figure size for better spacing

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


def plot_trajectory_split_by_alpha(hyperparam_to_paths: OrderedDict, output_filename: str,
                                   fig_title: str, split: str = 'valid') -> None:
    unique_alphas = sorted(set(a for a, _ in hyperparam_to_paths.keys()), reverse=True)
    unique_betas = sorted(set(b for _, b in hyperparam_to_paths.keys()))
    
    beta_colors = {
        -0.5: '#7b0000',
        0.0: '#e10000',
        0.5: '#ffc700'
    }

    fig, axes = plt.subplots(1, len(unique_alphas), figsize=(10, 3.5), sharex=True, sharey=True)

    for i, alpha in enumerate(unique_alphas):
        ax = axes[i]
        for (a, beta), file_path in hyperparam_to_paths.items():
            if a != alpha:
                continue
            with open(osp.join(file_path, f"metrics_over_time_{split}_mixed.pkl"), "rb") as f:
                total_metrics = p.load(f)

            pa = np.array(total_metrics['pairwise_accuracy']).mean(axis=0) * 100
            mrr = np.array(total_metrics['mrr_hard']).mean(axis=0)[1:] * 100

            color = beta_colors[beta]

            # Trajectory line
            ax.plot(pa, mrr, color=color, alpha=0.8, lw=2)
            
            # Normal circles for all timesteps except the first
            ax.scatter(pa[1:], mrr[1:], c=[color], edgecolors=color,
                       marker='o', s=30, linewidths=1.5, alpha=0.9)

            # Highlight first timestep
            ax.scatter(pa[0], mrr[0], marker='o', c=[color], s=30,
                       edgecolors='black', linewidths=1.5, zorder=10)

        ax.set_title(f'α = {alpha}', fontsize=13)
        ax.grid(alpha=0.3, linestyle='--')
        if i == 0:
            ax.set_ylabel('MRR (%)')
        ax.set_xlabel('Pairwise Accuracy (%)')
        
        if i == 0:
            legend_handles = [plt.Line2D([0], [0], color=beta_colors[b], lw=3, 
                                        label=f'β = {b}') for b in unique_betas]
            ax.legend(handles=legend_handles, loc='upper left', fontsize=10)

    fig.suptitle(fig_title, y=0.90)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    os.makedirs("plots", exist_ok=True)
    plt.savefig(osp.join("plots", output_filename), dpi=300, bbox_inches='tight')
    print(f"Saved plot to plots/{output_filename}")


# plot_metrics_comparison(
#     OrderedDict([
#         ("Unconstrained", "results/fb15k237/unconstrained/test/fb15k237-betae_10_0.0002_default_test_mixed_1760968758_073xj415"),
#         ("LightGBM", "results/fb15k237/lightgbm/test/fb15k237-betae_10_0.0002_lightgbm_lambdamart_test_mixed_1762078248_c2w6wno3"),
#         ("Cosine", "results/fb15k237/cosine/test/fb15k237-betae_10_0.0002_cosine_0.25_0.5_test_mixed_1761037498_fvwjmdjs"),
#         ("NQR", "results/fb15k237/ranknet/test/fb15k237-betae_10_0.0002_ranknet_0.001_test_mixed_1761053698_5fpk3jiw"),
#     ]),
#     output_filename="fb15k237.pdf",
#     fig_title="FB15k-237",
#     add_legend=False
# )

plot_metrics_comparison(
    OrderedDict([
        ("Unconstrained",
         "results/fb15k237/unconstrained/test-annotated/fb15k237-betae_10_0.0002_default_test_mixed_1773580568_8fklt3f3"),
        ("LightGBM",
         "results/fb15k237/lightgbm/test-annotated/fb15k237-betae_10_0.0002_lightgbm_lambdamart_test_mixed_1773580922_94ffp3bn"),
        ("Cosine",
         "results/fb15k237/cosine/test-annotated/fb15k237-betae_10_0.0002_cosine_0.25_0.5_test_mixed_1773580653_myewf0el"),
        ("NQR",
         "results/fb15k237/ranknet/test-annotated/fb15k237-betae_10_0.0002_ranknet_0.001_test_mixed_1773581120_2gj4rjkl"),
    ]),
    output_filename="fb15k237-annotated.pdf",
    fig_title="Annotated FB15k-237",
    add_legend=True
)

# plot_metrics_comparison(
#     OrderedDict([
#         ("Unconstrained", "results/hetionet/unconstrained/test/hetionet_10_0.001_default_test_mixed_1760975302_nblg4ymc"),
#         ("LightGBM", "results/hetionet/lightgbm/test/hetionet_10_0.001_lightgbm_lambdamart_test_mixed_1762163114_n3ae3wbh"),
#         ("Cosine", "results/hetionet/cosine/test/hetionet_10_0.001_cosine_0.25_0.5_test_mixed_1761037524_0fys9umr"),
#         ("NQR", "results/hetionet/ranknet/test/hetionet_10_0.001_ranknet_0.001_test_mixed_1761056339_3ekqazqe")
#     ]),
#     output_filename="hetionet.pdf",
#     fig_title="Hetionet",
#     add_legend=True
# )
#
# # Trajectory plot: PA vs MRR over time for each hyperparameter combination (FB15k-237)
# plot_trajectory_split_by_alpha(
#     OrderedDict([
#         ((0.25, -0.5), "results/fb15k237/cosine/valid/fb15k237-betae_10_0.0002_cosine_0.25_-0.5_valid_mixed_1760977590_ck0kmwly"),
#         ((0.25, 0.0), "results/fb15k237/cosine/valid/fb15k237-betae_10_0.0002_cosine_0.25_0.0_valid_mixed_1760977591_0lp8c2o1"),
#         ((0.25, 0.5), "results/fb15k237/cosine/valid/fb15k237-betae_10_0.0002_cosine_0.25_0.5_valid_mixed_1760977604_bvswxmfr"),
#         ((0.5, -0.5), "results/fb15k237/cosine/valid/fb15k237-betae_10_0.0002_cosine_0.5_-0.5_valid_mixed_1760977599_5k5u9uds"),
#         ((0.5, 0.0), "results/fb15k237/cosine/valid/fb15k237-betae_10_0.0002_cosine_0.5_0.0_valid_mixed_1760977597_hvvuw4cg"),
#         ((0.5, 0.5), "results/fb15k237/cosine/valid/fb15k237-betae_10_0.0002_cosine_0.5_0.5_valid_mixed_1760977599_j767jdpi"),
#         ((0.75, -0.5), "results/fb15k237/cosine/valid/fb15k237-betae_10_0.0002_cosine_0.75_-0.5_valid_mixed_1760977608_6z6x9tat"),
#         ((0.75, 0.0), "results/fb15k237/cosine/valid/fb15k237-betae_10_0.0002_cosine_0.75_0.0_valid_mixed_1760977602_lsil13w3"),
#         ((0.75, 0.5), "results/fb15k237/cosine/valid/fb15k237-betae_10_0.0002_cosine_0.75_0.5_valid_mixed_1760977608_9pkg2phd"),
#     ]),
#     output_filename="fb15k237_cosine_trajectory.pdf",
#     fig_title="FB15k-237",
#     split='valid'
# )
#
# # Trajectory plot: PA vs MRR over time for each hyperparameter combination (Hetionet)
# plot_trajectory_split_by_alpha(
#     OrderedDict([
#         ((0.25, -0.5), "results/hetionet/cosine/valid/hetionet_10_0.001_cosine_0.25_-0.5_valid_mixed_1760977644_f30p5v18"),
#         ((0.25, 0.0), "results/hetionet/cosine/valid/hetionet_10_0.001_cosine_0.25_0.0_valid_mixed_1760977644_u03ru7b6"),
#         ((0.25, 0.5), "results/hetionet/cosine/valid/hetionet_10_0.001_cosine_0.25_0.5_valid_mixed_1760977647_ujf8a0vz"),
#         ((0.5, -0.5), "results/hetionet/cosine/valid/hetionet_10_0.001_cosine_0.5_-0.5_valid_mixed_1760977647_zy2dhuqu"),
#         ((0.5, 0.0), "results/hetionet/cosine/valid/hetionet_10_0.001_cosine_0.5_0.0_valid_mixed_1760977651_dezss4c8"),
#         ((0.5, 0.5), "results/hetionet/cosine/valid/hetionet_10_0.001_cosine_0.5_0.5_valid_mixed_1760977647_i3p0imm0"),
#         ((0.75, -0.5), "results/hetionet/cosine/valid/hetionet_10_0.001_cosine_0.75_-0.5_valid_mixed_1760977653_q835wxra"),
#         ((0.75, 0.0), "results/hetionet/cosine/valid/hetionet_10_0.001_cosine_0.75_0.0_valid_mixed_1760977649_6226hm3d"),
#         ((0.75, 0.5), "results/hetionet/cosine/valid/hetionet_10_0.001_cosine_0.75_0.5_valid_mixed_1760977654_y4vqplf6"),
#     ]),
#     output_filename="hetionet_cosine_trajectory.pdf",
#     fig_title="Hetionet",
#     split='valid'
# )
