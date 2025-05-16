from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pickle as p
import itertools
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import LogNorm

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
    x_start = [1, 1, 1]
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


# plot_metrics_comparison(OrderedDict([
#     ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_1746902312_mcxg6hoy/metrics_over_time_test_mixed.pkl"),
#     ("Cosine", "results/fb15k237-betae_10_0.0002_cosine_0.1_0.9_test_mixed_1746902660_jsqnodzr/metrics_over_time_test_mixed.pkl"),
#     ("Ranknet", "results/fb15k237-betae_10_0.0002_ranknet_test_mixed_1747160013_c610piag/metrics_over_time_test_mixed.pkl"),
#     ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_1746903805_34ftyrot/metrics_over_time_test_mixed.pkl")
#     ]),
#     output_filename="fb15k237_over_time.pdf"
# )

# plot_metrics_comparison(OrderedDict([
#     ("QTO", "results/hetionet_10_0.001_default_test_mixed_1746908572_wlxyi0d7/metrics_over_time_test_mixed.pkl"),
#     ("Cosine", "results/hetionet_10_0.001_cosine_0.1_0.9_test_mixed_1746909609_14rps2d8/metrics_over_time_test_mixed.pkl"),
#     ("RankNet", "results/hetionet_10_0.001_ranknet_test_mixed_1747222015_y13wghbh/metrics_over_time_test_mixed.pkl"),
#     ("NQR", "results/hetionet_10_0.001_nqr_0.001_test_mixed_1747027748_dj60prrp/metrics_over_time_test_mixed.pkl")
#     ]),
#     output_filename="hetionet_over_time.pdf"
# )

def merge_test_and_sweep_data(test_runs_path, sweep_runs_path, output_path):
    test_runs_data = pd.read_csv(test_runs_path)
    sweep_runs_data = pd.read_csv(sweep_runs_path)
    mrr, acc, ids = [], [], []
    for path, checkpoint in tqdm(zip(test_runs_data["output_path"], test_runs_data["checkpoint"])):
        model_id = checkpoint.split("/")[-1].split("-")[0]
        ids.append(model_id)
        with open(osp.join(path, 'metrics_over_time_test_mixed.pkl'), 'rb') as f:
            metrics_over_time = p.load(f)

        final_mrr = np.array(metrics_over_time['mrr_hard'])[:,-1].mean()
        final_acc = np.array(metrics_over_time['pairwise_accuracy'])[:,-1].mean()
        mrr.append(final_mrr)
        acc.append(final_acc)

    final_performance_df = pd.DataFrame({"ID": ids, "MRR": mrr, "Accuracy": acc})
    final_performance_with_kl = pd.merge(final_performance_df, sweep_runs_data, on="ID", how="inner")

    final_performance_with_kl.to_csv(output_path)
    print(f"Saved {output_path}")


# merge_test_and_sweep_data("results/wandb_export_fb15k237_test_runs.csv",
#                           "results/wandb_export_fb15k237_sweep_runs.csv",
#                           "results/fb15k237_final_performance_with_kl.csv"))
# merge_test_and_sweep_data("results/wandb_export_hetionet_test_runs.csv",
#                           "results/wandb_export_hetionet_sweep_runs.csv",
#                           "results/hetionet_final_performance_with_kl.csv")


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_performance_with_kl(merged_data_paths, titles, highlights=None):
    num_plots = len(merged_data_paths)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

    # Handle the case where there's only one plot (axes might not be iterable)
    if num_plots == 1:
        axes = [axes]

    # Store handles for the legend (use a dictionary to avoid duplicates)
    legend_handles = {}

    # Iterate over each file path and create a subplot
    for i, path in enumerate(merged_data_paths):
        dataframe = pd.read_csv(path)

        # Scatter plot with color indicating KL weight
        scatter = axes[i].scatter(dataframe['Accuracy'], dataframe['MRR'],
                                  c=dataframe['kl_weight'], cmap='PuRd',
                                  norm=LogNorm(vmin=0.1, vmax=10), s=50,
                                  alpha=0.7, marker='o', edgecolors='black')

        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Pairwise Accuracy")
        axes[i].set_ylabel("MRR")

        # Plot highlight markers if provided
        if highlights:
            for highlight in highlights:
                x, y = highlight['coords'][i]
                marker = axes[i].scatter(x, y, color=highlight['color'],
                                         label=highlight['label'], s=200,
                                         marker='X', edgecolors='white', linewidths=1.5)
                # Collect handles in a dictionary to prevent duplicates
                if highlight['label'] not in legend_handles:
                    legend_handles[highlight['label']] = marker

    # Add a single colorbar on the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("KL Weight")
    cbar.set_ticks([0.1, 1.0, 10])
    cbar.set_ticklabels(['0.1', '1.0', '10'])

    # Add the legend for highlighted markers
    if legend_handles:
        fig.legend(handles=legend_handles.values(), loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=len(legend_handles))

    plt.show()

    
plot_performance_with_kl(["results/fb15k237_final_performance_with_kl.csv",
                          "results/hetionet_final_performance_with_kl.csv"],
                         ("FB15k237", "Hetionet"),
                         highlights=[
                             {"coords": [(0.5467, 0.1406), (0.4980, 0.0870)], "label": "QTO", "color": "C0"},
                             {"coords": [(0.9005, 0.0301), (0.7661, 0.0061)], "label": "Cosine", "color": "C1"},
                            {"coords": [(0.7547, 0.0821), (0.7139, 0.0686)], "label": "RankNet", "color": "C2"}
                         ]
                         )