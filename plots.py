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
        hits_at_1 = np.array(total_metrics_over_t_10['hits@1_hard'])
        hits_at_10 = np.array(total_metrics_over_t_10['hits@10_hard'])
        pairwise_acc = np.array(total_metrics_over_t_10['pairwise_accuracy'])
        ndcg_at_10 = np.array(total_metrics_over_t_10['ndcg@10'])
        ndcg_at_100 = np.array(total_metrics_over_t_10['ndcg@100'])

        metrics_all_methods.append([pairwise_acc, mrr_hard, hits_at_1, hits_at_10, ndcg_at_10, ndcg_at_100])

    # Create subplots for boxplots
    titles = ['Pairwise Accuracy', 'MRR', 'H@1', 'H@10','NDCG@10', 'NDCG@100']
    x_start = [1, 1, 1, 1, 1, 1]
    fig, axes = plt.subplots(1, len(titles), figsize=(3 * len(titles), 3.5))  # Increase figure size for better spacing

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

    # Collect first/last values for each method and metric
    summary_data = []
    method_list = []
    for method_idx, (metrics, method_name) in enumerate(zip(metrics_all_methods, method_names)):
        row = []
        for metric_idx, title in enumerate(titles):
            metric = metrics[metric_idx]
            if title == "Pairwise Accuracy":
                values = metric.mean(axis=0)
            else:
                values = metric.mean(axis=0)[1:]
            row.append(f"{values[-1]:.2f}")
        summary_data.append(row)
        method_list.append(method_name)

    df = pd.DataFrame(summary_data, columns=titles, index=method_list)
    print("\nSummary table (first/last for each metric):")
    print(df.to_markdown())

    for ax, title, metric_idx in zip(axes, titles, range(len(titles))):
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


# plot_metrics_comparison(OrderedDict([
#     ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_1753787312_xfap9fmm/metrics_over_time_test_mixed.pkl"),
#     ("Cosine", "results/fb15k237-betae_10_0.0002_cosine_0.1_0.9_test_mixed_1753788498_kap9hof5/metrics_over_time_test_mixed.pkl"),
#     ("Cosine-mean", "results/fb15k237-betae_10_0.0002_cosine_mean_test_mixed_1753808561_mttz7eyf/metrics_over_time_test_mixed.pkl"),
#     ("RankNet", "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_1753866499_meiyxexn/metrics_over_time_test_mixed.pkl"),
#     ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_1753815886_7l1mtfux/metrics_over_time_test_mixed.pkl")
# ]),
# output_filename="fb15k237_rebuttal.pdf"
# )

plot_metrics_comparison(OrderedDict([
    ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_1759243768_ylxe4zps/metrics_over_time_test_mixed.pkl"),
    # ("MeanCosine", "results/fb15k237-betae_10_0.0002_cosine_mean_test_mixed_1759244240_tomuhaj1/metrics_over_time_test_mixed.pkl"),
    # ("Cosine-mean", "results/fb15k237-betae_10_0.0002_cosine_mean_test_mixed_1759246262_5w14ugfg/metrics_over_time_test_mixed.pkl"),
    # ("Cosine-mean-f", "results/fb15k237-betae_10_0.0002_cosine_mean_test_mixed_1759246436_fibvuxuv/metrics_over_time_test_mixed.pkl"),
    ("MeanCosine-0.5-0.5", "results/fb15k237-betae_10_0.0002_cosine_mean_test_mixed_1759247239_nqyyi7ts/metrics_over_time_test_mixed.pkl"),
    ("NQR-sigm", "results/fb15k237-betae_10_0.0002_nqr_0.001_mixed_1760798240_1iyp5vi0/metrics_over_time_test_mixed.pkl"),
    # ("NQR-sigm-long", "results/fb15k237-betae_10_0.0002_nqr_0.001_mixed_1760798607_oyafeyma/metrics_over_time_test_mixed.pkl"),
    ("NQR-kl-1e-3", "results/fb15k237-betae_10_0.0002_nqr_0.001_mixed_1760800405_zv2w1wdv/metrics_over_time_test_mixed.pkl")
    # ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_mixed_1760772705_6lc19w8z/metrics_over_time_test_mixed.pkl"),
    # ("NQR-init", "results/fb15k237-betae_10_0.0002_nqr_0.001_mixed_1760777657_j06e0bu9/metrics_over_time_test_mixed.pkl"),
    # ("NQR-init-lowlr", "results/fb15k237-betae_10_0.0002_nqr_1e-09_mixed_1760777832_12ka497e/metrics_over_time_test_mixed.pkl"),
    # ("NQR-init-lowlr-10e", "results/fb15k237-betae_10_0.0002_nqr_1e-09_mixed_1760779573_4fc9vrn6/metrics_over_time_test_mixed.pkl")
]),
output_filename="fb15k237_1p.pdf"
)

# plot_metrics_comparison(OrderedDict([
#     ("QTO", "results/hetionet_10_0.001_default_test_mixed_1753881598_v92ke4vl/metrics_over_time_test_mixed.pkl"),
#     ("Cosine", "results/hetionet_10_0.001_cosine_0.1_0.9_test_mixed_1753909238_efyics4q/metrics_over_time_test_mixed.pkl"),
#     ("NQR", "results/hetionet_10_0.001_nqr_0.001_test_mixed_1753948455_pciziqbe/metrics_over_time_test_mixed.pkl")
# ]),
# output_filename="hetionet_rebuttal.pdf"
# )

# /home/daniel/projects/quack/results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_1753815886_7l1mtfux

# def merge_test_and_sweep_data(test_runs_path, sweep_runs_path, output_path):
#     test_runs_data = pd.read_csv(test_runs_path)
#     sweep_runs_data = pd.read_csv(sweep_runs_path)
#     mrr, acc, ids = [], [], []
#     for path, checkpoint in tqdm(zip(test_runs_data["output_path"], test_runs_data["checkpoint"])):
#         model_id = checkpoint.split("/")[-1].split("-")[0]
#         ids.append(model_id)
#         with open(osp.join(path, 'metrics_over_time_test_mixed.pkl'), 'rb') as f:
#             metrics_over_time = p.load(f)
#
#         final_mrr = np.array(metrics_over_time['mrr_hard'])[:,-1].mean()
#         final_acc = np.array(metrics_over_time['pairwise_accuracy'])[:,-1].mean()
#         mrr.append(final_mrr)
#         acc.append(final_acc)
#
#     final_performance_df = pd.DataFrame({"ID": ids, "MRR": mrr, "Accuracy": acc})
#     final_performance_with_kl = pd.merge(final_performance_df, sweep_runs_data, on="ID", how="inner")
#
#     final_performance_with_kl.to_csv(output_path)
#     print(f"Saved {output_path}")


# merge_test_and_sweep_data("results/wandb_export_fb15k237_test_runs.csv",
#                           "results/wandb_export_fb15k237_sweep_runs.csv",
#                           "results/fb15k237_final_performance_with_kl.csv"))
# merge_test_and_sweep_data("results/wandb_export_hetionet_test_runs.csv",
#                           "results/wandb_export_hetionet_sweep_runs.csv",
#                           "results/hetionet_final_performance_with_kl.csv")


# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from matplotlib.lines import Line2D
#
#
# def plot_performance_with_kl(merged_data_paths, titles, highlights=None):
#     num_plots = len(merged_data_paths)
#     fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
#
#     # Handle the case where there's only one plot (axes might not be iterable)
#     if num_plots == 1:
#         axes = [axes]
#
#     # Store handles for the legend (manually create legend entries)
#     legend_handles = {}
#
#     # Iterate over each file path and create a subplot
#     for i, path in enumerate(merged_data_paths):
#         dataframe = pd.read_csv(path)
#
#         # Scatter plot with color indicating KL weight
#         scatter = axes[i].scatter(dataframe['Accuracy'], dataframe['MRR'],
#                                   c=dataframe['kl_weight'], cmap='Reds',
#                                   norm=LogNorm(vmin=0.1, vmax=10), s=50,
#                                   alpha=0.7, marker='o', edgecolors='black')
#
#         axes[i].set_title(titles[i])
#         axes[i].set_xlabel("Pairwise Accuracy")
#         axes[i].set_ylabel("MRR")
#         axes[i].grid()
#
#         # Plot highlight markers if provided
#         if highlights:
#             for highlight in highlights:
#                 x, y = highlight['coords'][i]
#                 marker = axes[i].scatter(x, y, color=highlight['color'],
#                                          s=200, marker='X', edgecolors='white', linewidths=1.5)
#
#                 # Manually create a legend handle if not already present
#                 if highlight['label'] not in legend_handles:
#                     legend_handles[highlight['label']] = Line2D([0], [0], marker='X', color='w',
#                                                                 markerfacecolor=highlight['color'],
#                                                                 markersize=10, label=highlight['label'])
#
#     # Add a single colorbar on the right
#     fig.subplots_adjust(right=0.85)
#     cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#     cbar = fig.colorbar(scatter, cax=cbar_ax)
#     cbar.set_label("KL Weight")
#     cbar.set_ticks([0.1, 1.0, 10])
#     cbar.set_ticklabels(['0.1', '1.0', '10'])
#
#     # Add the legend for highlighted markers at the figure level
#     if legend_handles:
#         fig.legend(handles=legend_handles.values(), loc='lower center', bbox_to_anchor=(0.5, 0.05),
#                    ncol=len(legend_handles))
#     plt.subplots_adjust(bottom=0.275)
#     plt.savefig("mrr_accuracy_tradeoff.pdf")
#
#
# # plot_performance_with_kl("results/fb15k237_final_performance_with_kl.csv")
# plot_performance_with_kl(["results/fb15k237_final_performance_with_kl.csv",
#                           "results/hetionet_final_performance_with_kl.csv"],
#                          ("FB15k237", "Hetionet"),
#                          highlights=[
#                              {"coords": [(0.5467, 0.1406), (0.4980, 0.0870)], "label": "QTO", "color": "black"},
#                              {"coords": [(0.9005, 0.0301), (0.7661, 0.0061)], "label": "Cosine", "color": "C1"},
#                             {"coords": [(0.7547, 0.0821), (0.7139, 0.0686)], "label": "RankNet", "color": "C2"}
#                          ]
#                          )