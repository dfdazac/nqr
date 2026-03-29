from collections import OrderedDict

import pandas as pd
import os.path as osp

from nqr.qto.query import all_tasks


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


def print_latex_rows(df):
    """Print LaTeX formatted rows with method name and values for each query type.
    Bold the highest value in each column.
    
    Args:
        df: DataFrame returned by process_files with columns for 'method' and query types
    """
    type_columns = set(df.columns)
    type_columns.remove('method')
    tasks = [t for t in all_tasks if t in type_columns]
    tasks.append('avg')
    
    # Find the maximum value for each task column
    max_values = {}
    for task in tasks:
        max_values[task] = df[task].astype(float).max()
    
    print(f"Method & {' & '.join(tasks)} \\\\")
    for _, row in df.iterrows():
        method = row["method"]
        # Get values for each query type in the order of all_tasks
        formatted_values = []
        for task in tasks:
            value_str = row[task]
            value_float = float(value_str)
            # Bold the value if it's the maximum for this column
            if value_float == max_values[task]:
                formatted_values.append(f"\\bf {value_str}")
            else:
                formatted_values.append(value_str)
        # Create LaTeX row: Method & val1 & val2 & ... \\
        latex_row = f"{method} & " + " & ".join(formatted_values) + " \\\\"
        print(latex_row)


fb15k237_results = OrderedDict([
    ("Unconstrained", "results/fb15k237/unconstrained/test/fb15k237-betae_10_0.0002_default_test_mixed_1760968758_073xj415"),
    ("LightGBM", "results/fb15k237/lightgbm/test/fb15k237-betae_10_0.0002_lightgbm_lambdamart_test_mixed_1762078248_c2w6wno3"),
    ("Cosine", "results/fb15k237/cosine/test/fb15k237-betae_10_0.0002_cosine_0.25_0.5_test_mixed_1761037498_fvwjmdjs"),
    ("NQR", "results/fb15k237/ranknet/test/fb15k237-betae_10_0.0002_ranknet_0.001_test_mixed_1761053698_5fpk3jiw"),
])

print("=== FB15k237 ===")
print("Cumulative Pairwise Accuracy:")
print_latex_rows(process_files(fb15k237_results, "cumulative_pairwise_accuracy"))
print("Cumulative MRR Hard:")
print_latex_rows(process_files(fb15k237_results, "cumulative_mrr_hard"))
print("Cumulative NDCG@10:")
print_latex_rows(process_files(fb15k237_results, "cumulative_ndcg@10"))

fb15k237_annotated_results = OrderedDict([
    ("Unconstrained", "results/fb15k237/unconstrained/test-annotated/fb15k237-betae_10_0.0002_default_test_mixed_1773580568_8fklt3f3"),
    ("LightGBM", "results/fb15k237/lightgbm/test-annotated/fb15k237-betae_10_0.0002_lightgbm_lambdamart_test_mixed_1773580922_94ffp3bn"),
    ("Cosine", "results/fb15k237/cosine/test-annotated/fb15k237-betae_10_0.0002_cosine_0.25_0.5_test_mixed_1773580653_myewf0el"),
    ("NQR", "results/fb15k237/ranknet/test-annotated/fb15k237-betae_10_0.0002_ranknet_0.001_test_mixed_1773581120_2gj4rjkl"),
])

print("=== FB15k237 Annotated ===")
print("Cumulative Pairwise Accuracy:")
print_latex_rows(process_files(fb15k237_annotated_results, "cumulative_pairwise_accuracy"))
print("Cumulative MRR Hard:")
print_latex_rows(process_files(fb15k237_annotated_results, "cumulative_mrr_hard"))
print("Cumulative NDCG@10:")
print_latex_rows(process_files(fb15k237_annotated_results, "cumulative_ndcg@10"))

hetionet_results = OrderedDict([
    ("Unconstrained", "results/hetionet/unconstrained/test/hetionet_10_0.001_default_test_mixed_1760975302_nblg4ymc"),
    ("LightGBM", "results/hetionet/lightgbm/test/hetionet_10_0.001_lightgbm_lambdamart_test_mixed_1762163114_n3ae3wbh"),
    ("Cosine", "results/hetionet/cosine/test/hetionet_10_0.001_cosine_0.25_0.5_test_mixed_1761037524_0fys9umr"),
    ("NQR", "results/hetionet/ranknet/test/hetionet_10_0.001_ranknet_0.001_test_mixed_1761056339_3ekqazqe")
])

print("\n=== Hetionet ===")
print("Cumulative Pairwise Accuracy:")
print_latex_rows(process_files(hetionet_results, "cumulative_pairwise_accuracy"))
print("Cumulative MRR Hard:")
print_latex_rows(process_files(hetionet_results, "cumulative_mrr_hard"))
print("Cumulative NDCG@10:")
print_latex_rows(process_files(hetionet_results, "cumulative_ndcg@10"))

