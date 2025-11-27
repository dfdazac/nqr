from collections import OrderedDict
import os
import os.path as osp
import glob
import pandas as pd
from scipy import stats


def process_runtime_files(file_dict):
    """Process execution time files from directories.
    
    Args:
        file_dict: OrderedDict mapping method names to directory paths
        
    Returns:
        DataFrame with columns: method, avg_time_ms, delta_ms
    """
    data = []
    unconstrained_time = None
    
    for method, directory in file_dict.items():
        # Find execution_times_test_mixed*.txt file in the directory
        pattern = osp.join(directory, "execution_times_test_mixed*.txt")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print(f"Warning: No execution_times_test_mixed file found in {directory}")
            continue
        
        # Use the first matching file
        filepath = matching_files[0]
        
        # Parse the file to extract average time
        avg_time = None
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("Average time:"):
                    # Parse "Average time: 0.001819 seconds"
                    parts = line.strip().split()
                    avg_time = float(parts[2])  # Get the numeric value
                    break
        
        if avg_time is None:
            print(f"Warning: Could not find average time in {filepath}")
            continue
        
        # Convert to milliseconds
        avg_time_ms = avg_time * 1000
        
        # Store unconstrained time for delta calculation
        if method == "Unconstrained":
            unconstrained_time = avg_time_ms
        
        data.append({
            "method": method,
            "avg_time_ms": avg_time_ms
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate delta relative to Unconstrained
    if unconstrained_time is not None:
        df["delta_ms"] = df["avg_time_ms"] - unconstrained_time
    else:
        df["delta_ms"] = 0.0
    
    return df


def print_runtime_table(df):
    """Print a formatted table of runtime results.
    
    Args:
        df: DataFrame with columns method, avg_time_ms, delta_ms
    """
    print(f"{'Method':<20} {'Avg Time (ms)':<15} {'Delta (ms)':<15}")
    print("-" * 50)
    
    for _, row in df.iterrows():
        method = row["method"]
        avg_time = row["avg_time_ms"]
        delta = row["delta_ms"]
        
        # Format delta with + sign for positive values
        delta_str = f"{delta:+.1f}" if delta != 0 else "---"
        
        print(f"{method:<20} & {avg_time:<15.1f} & {delta_str:<15}\\\\")


def get_individual_times(directory):
    """Extract individual execution times from a directory.
    
    Args:
        directory: Path to directory containing execution_times_test_mixed*.txt
        
    Returns:
        List of execution times in seconds
    """
    pattern = osp.join(directory, "execution_times_test_mixed*.txt")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        return []
    
    filepath = matching_files[0]
    times = []
    
    # Flag to indicate we're in the individual times section
    in_times_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("Individual execution times"):
                in_times_section = True
                continue
            
            if in_times_section:
                line = line.strip()
                if ':' in line:
                    # Parse lines like "1: 0.003135"
                    parts = line.split(':')
                    if len(parts) == 2:
                        try:
                            time_value = float(parts[1].strip())
                            times.append(time_value)
                        except ValueError:
                            continue
    
    return times


def test_significance(method1_dir, method2_dir, method1_name, method2_name):
    """Perform paired t-test between two methods.
    
    Args:
        method1_dir: Directory for first method
        method2_dir: Directory for second method
        method1_name: Name of first method
        method2_name: Name of second method
    """
    times1 = get_individual_times(method1_dir)
    times2 = get_individual_times(method2_dir)
    
    if not times1 or not times2:
        print(f"Could not load individual times for comparison")
        return
    
    if len(times1) != len(times2):
        print(f"Warning: Different number of samples ({len(times1)} vs {len(times2)})")
        # Take minimum length for paired test
        min_len = min(len(times1), len(times2))
        times1 = times1[:min_len]
        times2 = times2[:min_len]
    
    # Convert to milliseconds for display
    times1_ms = [t * 1000 for t in times1]
    times2_ms = [t * 1000 for t in times2]
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(times1_ms, times2_ms)
    
    # Calculate means
    mean1 = sum(times1_ms) / len(times1_ms)
    mean2 = sum(times2_ms) / len(times2_ms)
    
    print(f"\n=== Statistical Significance Test: {method1_name} vs {method2_name} ===")
    print(f"Number of samples: {len(times1)}")
    print(f"{method1_name} mean: {mean1:.4f} ms")
    print(f"{method2_name} mean: {mean2:.4f} ms")
    print(f"Difference: {mean2 - mean1:.4f} ms")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    
    if p_value < 0.001:
        print(f"Result: Highly significant (p < 0.001) ***")
    elif p_value < 0.01:
        print(f"Result: Very significant (p < 0.01) **")
    elif p_value < 0.05:
        print(f"Result: Significant (p < 0.05) *")
    else:
        print(f"Result: Not significant (p >= 0.05)")


# FB15k237 results
fb15k237_results = OrderedDict([
    ("Unconstrained", "results/fb15k237/unconstrained/runtimes/fb15k237-betae_10_0.0002_default_test_mixed_1763459526_3w66efuv"),
    ("LightGBM", "results/fb15k237/lightgbm/runtimes/fb15k237-betae_10_0.0002_lightgbm_lambdamart_test_mixed_1763461575_4pzj6b85"),
    ("Cosine", "results/fb15k237/cosine/runtimes/fb15k237-betae_10_0.0002_cosine_0.25_0.5_test_mixed_1763458916_zy1xhymt"),
    ("NQR", "results/fb15k237/ranknet/runtimes/fb15k237-betae_10_0.0002_ranknet_0.01_test_mixed_1763463312_lqu142oi"),
])

print("=== FB15k237 Runtime Analysis ===")
df_fb15k237 = process_runtime_files(fb15k237_results)
print_runtime_table(df_fb15k237)

# Test significance between Cosine and NQR
test_significance(
    fb15k237_results["Cosine"],
    fb15k237_results["NQR"],
    "Cosine",
    "NQR"
)

