from collections import OrderedDict

from quack.qto.query import name_answer_dict

import os.path as osp
import pandas as pd

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

    # Convert list of dicts to a DataFrame
    df = pd.DataFrame(data)
    
    # Sort columns alphabetically, but keep 'method' first and 'avg' last
    cols = df.columns.tolist()
    method_col = ['method'] if 'method' in cols else []
    avg_col = ['avg'] if 'avg' in cols else []
    other_cols = sorted([col for col in cols if col not in ['method', 'avg']])
    df = df[method_col + other_cols + avg_col]
    
    return df

# TARGET MODE TABLES (4 methods: QTO, MeanCosine, NQR, SCORE)

# FB15K237 - Target Mode
print("=== FB15K237 TARGET MODE ===")
fb15k237_target_results = OrderedDict([
    ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_target_1757178216_rq8chcd6"),
    ("MeanCosine", "results/fb15k237-betae_10_0.0002_cosine_mean_test_mixed_target_1757240412_4fkuo5mv"),
    ("NQR", "results/fb15k237-betae_10_0.0002_nqr_0.001_test_mixed_target_1757332560_2s3z0uom"),
    ("SCORE", "results/fb15k237-betae_10_0.0002_score_test_mixed_target_1757349746_vcvn45r2")
])
print("Cumulative Pairwise Accuracy:")
print(process_files(fb15k237_target_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print("Cumulative MRR Hard:")
print(process_files(fb15k237_target_results, "cumulative_mrr_hard").to_latex(index=False))
print("Cumulative NDCG@10:")
print(process_files(fb15k237_target_results, "cumulative_ndcg@10").to_latex(index=False))

# HETIONET - Target Mode
print("\n=== HETIONET TARGET MODE ===")
hetionet_target_results = OrderedDict([
    ("QTO", "results/hetionet_10_0.001_default_test_mixed_target_1757178246_dxkau1zu"),
    ("MeanCosine", "results/hetionet_10_0.001_cosine_mean_test_mixed_target_1757240403_6n5n8deo"),
    ("NQR", "results/hetionet_10_0.001_nqr_0.001_test_mixed_target_1757350038_cd4q5jcy"),
    ("SCORE", "results/hetionet_10_0.001_score_test_mixed_target_1757349853_6opdbv9s")
])
print("Cumulative Pairwise Accuracy:")
print(process_files(hetionet_target_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print("Cumulative MRR Hard:")
print(process_files(hetionet_target_results, "cumulative_mrr_hard").to_latex(index=False))
print("Cumulative NDCG@10:")
print(process_files(hetionet_target_results, "cumulative_ndcg@10").to_latex(index=False))

# NELL995 - Target Mode
print("\n=== NELL995 TARGET MODE ===")
nell995_target_results = OrderedDict([
    ("QTO", "results/nell-betae_10_0.0002_default_test_mixed_target_1757178211_0pf4oo3l"),
    ("MeanCosine", "results/nell-betae_10_0.0002_cosine_mean_test_mixed_target_1757240402_p0iy5bvo"),
    ("NQR", "results/nell-betae_10_0.0002_nqr_0.001_test_mixed_target_1757350066_70pd7fnx"),
    ("SCORE", "results/nell-betae_10_0.0002_score_test_mixed_target_1757349914_mim7r1ox")
])
print("Cumulative Pairwise Accuracy:")
print(process_files(nell995_target_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print("Cumulative MRR Hard:")
print(process_files(nell995_target_results, "cumulative_mrr_hard").to_latex(index=False))
print("Cumulative NDCG@10:")
print(process_files(nell995_target_results, "cumulative_ndcg@10").to_latex(index=False))

# FULL MODE TABLES (2 methods: QTO, SCORE)

# FB15K237 - Full Mode
print("\n=== FB15K237 FULL MODE ===")
fb15k237_full_results = OrderedDict([
    ("QTO", "results/fb15k237-betae_10_0.0002_default_test_mixed_full_1757178217_q1f9v156"),
    ("SCORE", "results/fb15k237-betae_10_0.0002_score_test_mixed_full_1757349830_k9k9hl3w")
])
print("Cumulative Pairwise Accuracy:")
print(process_files(fb15k237_full_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print("Cumulative MRR Hard:")
print(process_files(fb15k237_full_results, "cumulative_mrr_hard").to_latex(index=False))
print("Cumulative NDCG@10:")
print(process_files(fb15k237_full_results, "cumulative_ndcg@10").to_latex(index=False))

# HETIONET - Full Mode
print("\n=== HETIONET FULL MODE ===")
hetionet_full_results = OrderedDict([
    ("QTO", "results/hetionet_10_0.001_default_test_mixed_full_1757178247_2vbjrw1k"),
    ("SCORE", "results/hetionet_10_0.001_score_test_mixed_full_1757349853_s6l28qfp")
])
print("Cumulative Pairwise Accuracy:")
print(process_files(hetionet_full_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print("Cumulative MRR Hard:")
print(process_files(hetionet_full_results, "cumulative_mrr_hard").to_latex(index=False))
print("Cumulative NDCG@10:")
print(process_files(hetionet_full_results, "cumulative_ndcg@10").to_latex(index=False))

# NELL995 - Full Mode
print("\n=== NELL995 FULL MODE ===")
nell995_full_results = OrderedDict([
    ("QTO", "results/nell-betae_10_0.0002_default_test_mixed_full_1757178218_9jqnj61o"),
    ("SCORE", "results/nell-betae_10_0.0002_score_test_mixed_full_1757349943_q3v0uj0e")
])
print("Cumulative Pairwise Accuracy:")
print(process_files(nell995_full_results, "cumulative_pairwise_accuracy").to_latex(index=False))
print("Cumulative MRR Hard:")
print(process_files(nell995_full_results, "cumulative_mrr_hard").to_latex(index=False))
print("Cumulative NDCG@10:")
print(process_files(nell995_full_results, "cumulative_ndcg@10").to_latex(index=False))
