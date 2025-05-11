from collections import OrderedDict

import pandas as pd

def process_files(file_dict, metric_name):
    data = []

    for method, file in file_dict.items():
        metrics = {}
        metrics["method"] = method
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
    ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_1746903805_34ftyrot/all_metrics_test_mixed.txt")
    ])

print(process_files(fb15k237_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print(process_files(fb15k237_results, "cumulative_mrr_hard").to_latex(index=False))
