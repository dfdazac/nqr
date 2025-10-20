from collections import OrderedDict

import pandas as pd
import os.path as osp


def process_files(file_dict, metric_name):
    data = []

    for method, file in file_dict.items():
        metrics = {}
        metrics["method"] = method

        all_values = []
        with open(osp.join(file, "all_metrics_test_mixed.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue  # Ignore malformed lines
                _, query_type, metric, value = parts
                value = float(value)
                if metric == f"{metric_name}:":
                    all_values.append(value)
                    metrics[f"{query_type}"] = f"{value * 100:.2f}"

        metrics["avg"] = f"{pd.Series(all_values).mean() * 100:.2f}"
        data.append(metrics)

    df = pd.DataFrame(data)

    cols = df.columns.tolist()
    method_col = ['method'] if 'method' in cols else []
    avg_col = ['avg'] if 'avg' in cols else []
    other_cols = sorted([col for col in cols if col not in ['method', 'avg']])
    df = df[method_col + other_cols + avg_col]

    return df

fb15k237_results = OrderedDict([
    ("Unconstrained", "results/fb15k237-betae_10_0.0002_default_test_mixed_1759243768_ylxe4zps"),
    ("Cosine", "results/fb15k237-betae_10_0.0002_cosine_mean_test_mixed_1759247239_nqyyi7ts"),
    ("Ranknet", "results/fb15k237-betae_10_0.0002_nqr_0.001_mixed_1760798240_1iyp5vi0"),
    ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_mixed_1760800405_zv2w1wdv")
    ])

print("Cumulative Pairwise Accuracy:")
print(process_files(fb15k237_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print("Cumulative MRR Hard:")
print(process_files(fb15k237_results, "cumulative_mrr_hard").to_latex(index=False))
print("Cumulative NDCG@10:")
print(process_files(fb15k237_results, "cumulative_ndcg@10").to_latex(index=False))
