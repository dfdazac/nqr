from collections import OrderedDict

from quack.qto.query import name_answer_dict

import pandas as pd

def process_files(file_dict, metric_name):
    data = []

    for method, file in file_dict.items():
        metrics = {}
        metrics["method"] = method
        for m in name_answer_dict:
            metrics[m] = None

        all_values = []
        with open(file, 'r') as f:
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

    # Convert list of dicts to a DataFrame
    df = pd.DataFrame(data)
    return df

fb15k237_results = OrderedDict([
    ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_1746902312_mcxg6hoy/all_metrics_test_mixed.txt"),
    ("Cosine", "results/fb15k237-betae_10_0.0002_cosine_0.1_0.9_test_mixed_1746902660_jsqnodzr/all_metrics_test_mixed.txt"),
    ("Ranknet", "results/fb15k237-betae_10_0.0002_ranknet_test_mixed_1747160013_c610piag/all_metrics_test_mixed.txt"),
    ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_1746903805_34ftyrot/all_metrics_test_mixed.txt")
    ])

print(process_files(fb15k237_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print(process_files(fb15k237_results, "cumulative_mrr_hard").to_latex(index=False))

hetionet_results = OrderedDict([
    ("QTO", "results/hetionet_10_0.001_default_test_mixed_1746908572_wlxyi0d7/all_metrics_test_mixed.txt"),
    ("Cosine", "results/hetionet_10_0.001_cosine_0.1_0.9_test_mixed_1746909609_14rps2d8/all_metrics_test_mixed.txt"),
    ("Ranknet", "results/hetionet_10_0.001_ranknet_test_mixed_1747222015_y13wghbh/all_metrics_test_mixed.txt"),
    ("NQR", "results/hetionet_10_0.001_nqr_0.001_test_mixed_1747027748_dj60prrp/all_metrics_test_mixed.txt")
    ])

print(process_files(hetionet_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print(process_files(hetionet_results, "cumulative_mrr_hard").to_latex(index=False))