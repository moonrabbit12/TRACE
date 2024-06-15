import os
import json
import re
import math
import sys
import numpy as np
import scipy

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats
from scipy.stats import sem, t
from scipy import stats


def save_performance_and_bwt_to_csv(tables, output_directory, extraname):
    data = []

    for config_key, table in tables.items():
        # Ensure metrics are calculated
        table = calculate_metrics(table)
        
        # Extract average performance and its confidence interval
        avg_performance_mean, avg_performance_ci_lower, avg_performance_ci_upper = table.loc['Avg Performance', table.columns[-1]]
        avg_performance_ci_margin = avg_performance_ci_upper - avg_performance_ci_lower
        
        # Extract BWT and its confidence interval
        bwt_mean, bwt_ci_lower, bwt_ci_upper = table.loc['BWT', table.columns[-1]]
        bwt_ci_margin = bwt_ci_upper - bwt_ci_lower

        # Append data
        data.append({
            'Configuration': config_key,
            'Average Performance Mean': avg_performance_mean,
            'Average Performance CI Lower': avg_performance_ci_lower,
            'Average Performance CI Upper': avg_performance_ci_upper,
            'Average Performance CI Margin': avg_performance_ci_margin,
            'BWT Mean': bwt_mean,
            'BWT CI Lower': bwt_ci_lower,
            'BWT CI Upper': bwt_ci_upper,
            'BWT CI Margin': bwt_ci_margin
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_csv_path = os.path.join(output_directory, extraname+'_performance_and_bwt.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"CSV saved to {output_csv_path}")



def save_plot_to_pdf_improved(data, output_directory, name, cfg, baselines):
    # Unpack cfg into respective variables
    print(cfg, name)
    if name == "dim_size":
        met, mod, dat, aff, ite, part, infer = cfg
        xname = 'weak dimension size'
    elif name == "step_size":
        met, mod, ite, dat, aff, part, infer = cfg
        xname = 'affine shift size'

    if met == 'svd_replay':
        met_name = "CLAP+Replay"
    elif met == 'PCA':
        met_name = 'CLAP'
    else:
        met_name = met
    # Extract iterations and outputs from the list of tuples
    iterations = [int(item[0]) for item in data]
    outputs = [item[1] for item in data]

    # Sort based on iterations
    paired_sorted = sorted(zip(iterations, outputs), key=lambda pair: pair[0])
    iterations, outputs = zip(*paired_sorted)

    averages, lower, upper = zip(*outputs)
    lower_errors = [avg - min_val for avg, min_val in zip(averages, lower)]
    upper_errors = [max_val - avg for avg, max_val in zip(averages, upper)]

    # Plot main data with confidence intervals
    plt.figure(figsize=(10, 6))
    # Plot error bars separately
    plt.errorbar(iterations, averages, yerr=[lower_errors, upper_errors], 
                 fmt='none', ecolor='lightblue', elinewidth=2, capsize=5, label='_nolegend_')
    # Plot data points separately with desired marker
    plt.plot(iterations, averages, 'o-', label=met_name)

    # For baseline error bars: plot each baseline as a horizontal line with error bars
    for baseline_name in ['base', 'replay', 'EWC', 'LFPT5', 'L2P']:
        baseline_key = (baseline_name, mod, dat)
        if baseline_key in baselines:
            # Extract baseline average, min, and max values
            baseline_avg, baseline_min, baseline_max = baselines[baseline_key]
            # Calculate errors for baseline
            baseline_avgs = [baseline_avg for i in range(len(iterations))]
            baseline_lower = baseline_avg - baseline_min
            baseline_upper = baseline_max - baseline_avg
            baseline_lowers = [baseline_lower for i in range(len(iterations))]
            baseline_uppers = [baseline_upper for i in range(len(iterations))]
            baseline_errors = [baseline_lowers, baseline_uppers]

            # Choose colors
            if baseline_name == 'base':
                baseline_label = 'Naive'
                color_main = 'orange'
                color_light = '#FFA07A'  # Light Orange
                marker = 's'
            elif baseline_name == 'replay':
                baseline_label = 'Replay'
                color_main = 'green'
                color_light = '#90EE90'  # Light Green
                marker = '^'
            elif baseline_name == 'EWC':
                baseline_label = 'EWC'
                color_main = 'red'
                color_light = '#FF7F7F'  # Light Red
                marker = 'D'
            elif baseline_name == 'LFPT5':
                baseline_label = baseline_name
                color_main = 'purple'
                color_light = '#D8BFD8'
                marker = '8'
            elif baseline_name == 'L2P':
                baseline_label = baseline_name
                color_main = '#FFC0CB'
                color_light = '#FFB6C1'
                marker = 'p'

            # Plot baseline error bars in light color
            plt.errorbar(iterations, baseline_avgs, yerr=baseline_errors,
                         fmt='none', ecolor=color_light, capsize=5, label='_nolegend_')
            # Plot baseline main line and markers in prominent color
            plt.plot(iterations, baseline_avgs, marker=marker, linestyle='-', color=color_main, markersize=8, label=f'{baseline_label}')
    if part == 'mha_only':
        plttitle = 'MHA Projection'
    elif part == 'ov_only':
        plttitle = 'OV Projection'
    elif part == 'qk_only':
        plttitle = 'QK Projection'
    elif part == 'ffn_only':
        plttitle = 'FFN Projection'
    elif part == 'full':
        plttitle = 'Full Projection'
        
    plt.title(plttitle, fontsize=14)
    plt.xlabel(xname, fontsize=12)
    plt.ylabel('performance', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust layout to make room for the legend and minimize margins
    plt.tight_layout()
    
    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)

    # Save the plot to a PDF
    cfg_name = '_'.join(str(item) for item in cfg)
    filename = f"{cfg_name}_{name}.pdf"
    path = os.path.join(output_directory, filename)
    with PdfPages(path) as pdf:
        pdf.savefig(bbox_inches='tight')
        plt.close()

    print(f"Plot saved to {path}")

def save_comparison_plot_to_pdf(data_traintestbase, data_traintestSVD, output_directory, cfg):
    # Unpack configuration
    met, mod, dat, aff, ite, part, infer = cfg
    
    # Extract and sort iterations and outputs for both datasets
    iterations_base = [int(item[0]) for item in data_traintestbase]
    outputs_base = [item[1] for item in data_traintestbase]
    paired_sorted_base = sorted(zip(iterations_base, outputs_base), key=lambda pair: pair[0])
    iterations_base, outputs_base = zip(*paired_sorted_base)
    averages_base, lower_base, upper_base = zip(*outputs_base)
    lower_errors_base = [avg - min_val for avg, min_val in zip(averages_base, lower_base)]
    upper_errors_base = [max_val - avg for avg, max_val in zip(averages_base, upper_base)]
    
    iterations_svd = [int(item[0]) for item in data_traintestSVD]
    outputs_svd = [item[1] for item in data_traintestSVD]
    paired_sorted_svd = sorted(zip(iterations_svd, outputs_svd), key=lambda pair: pair[0])
    iterations_svd, outputs_svd = zip(*paired_sorted_svd)
    averages_svd, lower_svd, upper_svd = zip(*outputs_svd)
    lower_errors_svd = [avg - min_val for avg, min_val in zip(averages_svd, lower_svd)]
    upper_errors_svd = [max_val - avg for avg, max_val in zip(averages_svd, upper_svd)]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.errorbar(iterations_base, averages_base, yerr=[lower_errors_base, upper_errors_base],
                 fmt='none', ecolor='lightblue', elinewidth=2, capsize=5, label='_nolegend_')
    plt.plot(iterations_base, averages_base, 'o-', label='traintestbase')
    
    plt.errorbar(iterations_svd, averages_svd, yerr=[lower_errors_svd, upper_errors_svd],
                 fmt='none', ecolor='lightgreen', elinewidth=2, capsize=5, label='_nolegend_')
    plt.plot(iterations_svd, averages_svd, 's-', label='traintestSVD')
    
    # Set titles and labels
    plt.title(f"Comparison of traintestbase and traintestSVD\n{met}, {mod}, {dat}, {aff}, {ite}, {part}, {infer}", fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the plot to a PDF
    cfg_name = '_'.join(str(item) for item in cfg)
    filename = f"{cfg_name}_comparison.pdf"
    path = os.path.join(output_directory, filename)
    with PdfPages(path) as pdf:
        pdf.savefig(bbox_inches='tight')
        plt.close()
    
    print(f"Comparison plot saved to {path}")

def extract_step_size_data(step_size_averages, inference_type):
    extracted_data = {}
    for cfg, values in step_size_averages.items():
        if cfg[-1] == inference_type:  # Check if the last element matches the inference type
            data = [(step_size, (avg, lower, upper)) for (step_size, (avg, lower, upper)) in values]
            extracted_data[cfg] = sorted(data, key=lambda x: x[0])  # Sort by step size
    return extracted_data

def save_comparison_plot_to_pdf(data_traintestbase, data_traintestSVD, output_directory, cfg):
    # Unpack configuration
    met, mod, dim, dat, aff, part, infer = cfg
    
    # Extract and sort iterations and outputs for both datasets
    iterations_base = [int(item[0]) for item in data_traintestbase]
    outputs_base = [item[1] for item in data_traintestbase]
    paired_sorted_base = sorted(zip(iterations_base, outputs_base), key=lambda pair: pair[0])
    iterations_base, outputs_base = zip(*paired_sorted_base)
    averages_base, lower_base, upper_base = zip(*outputs_base)
    lower_errors_base = [avg - min_val for avg, min_val in zip(averages_base, lower_base)]
    upper_errors_base = [max_val - avg for avg, max_val in zip(averages_base, upper_base)]
    
    iterations_svd = [int(item[0]) for item in data_traintestSVD]
    outputs_svd = [item[1] for item in data_traintestSVD]
    paired_sorted_svd = sorted(zip(iterations_svd, outputs_svd), key=lambda pair: pair[0])
    iterations_svd, outputs_svd = zip(*paired_sorted_svd)
    averages_svd, lower_svd, upper_svd = zip(*outputs_svd)
    lower_errors_svd = [avg - min_val for avg, min_val in zip(averages_svd, lower_svd)]
    upper_errors_svd = [max_val - avg for avg, max_val in zip(averages_svd, upper_svd)]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.errorbar(iterations_base, averages_base, yerr=[lower_errors_base, upper_errors_base],
                 fmt='none', ecolor='lightblue', elinewidth=2, capsize=5, label='_nolegend_')
    plt.plot(iterations_base, averages_base, 'o-', label='traintestbase')
    
    plt.errorbar(iterations_svd, averages_svd, yerr=[lower_errors_svd, upper_errors_svd],
                 fmt='none', ecolor='lightgreen', elinewidth=2, capsize=5, label='_nolegend_')
    plt.plot(iterations_svd, averages_svd, 's-', label='traintestSVD')
    
    # Set titles and labels
    plt.title(f"Comparison of traintestbase and traintestSVD\n{met}, {mod}, {dim} , {dat}, {aff}, {part}, {infer}", fontsize=14)
    plt.xlabel('Step Size', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the plot to a PDF
    cfg_name = '_'.join(str(item) for item in cfg)
    filename = f"{cfg_name}_comparison.pdf"
    path = os.path.join(output_directory, filename)
    with PdfPages(path) as pdf:
        pdf.savefig(bbox_inches='tight')
        plt.close()
    
    print(f"Comparison plot saved to {path}")


def save_all_affine_shift_plots_to_pdf(data_dict, output_directory, baselines):
    """
    Plot all ablations in one plot for each inference strategy and save them as a single PDF.

    Parameters:
    - data_dict: Dictionary containing all data to plot. Keys are tuples of configurations, values are data lists.
    - output_directory: Directory to save the PDF.
    - baselines: Dictionary of baseline data.
    """
    # Define the ablations and inference strategies
    ablations = ['mha_only', 'ov_only', 'qk_only', 'ffn_only', 'full']
    inference_strategies = ['base', 'SVD', 'SVDreal']

    # Extracting common parameters for the plot title from the first key
    cfg_common = None
    for cfg in data_dict.keys():
        cfg_common = cfg[:4]
        break

    cfg_common_str = ', '.join(cfg_common)

    # Define distinct colors and markers for each ablation
    ablation_colors = {
        'mha_only': 'blue',
        'ov_only': 'orange',
        'qk_only': 'green',
        'ffn_only': 'red',
        'full': 'purple'
    }
    ablation_markers = {
        'mha_only': 'o',
        'ov_only': 's',
        'qk_only': '^',
        'ffn_only': 'D',
        'full': 'P'
    }

    # Create the PDF file
    pdf_filename = f"all_affine_shift_plots_{cfg_common_str}.pdf"
    pdf_path = os.path.join(output_directory, pdf_filename)

    with PdfPages(pdf_path) as pdf:
        for strategy in inference_strategies:
            plt.figure(figsize=(14, 8))

            for ablation in ablations:
                cfg_key = tuple(list(cfg_common) + ['affine', ablation, strategy])
                if cfg_key in data_dict:
                    data = data_dict[cfg_key]
                    method = f"{ablation} ({strategy})"
                    iterations = [int(item[0]) for item in data]
                    outputs = [item[1] for item in data]

                    # Sort based on iterations
                    paired_sorted = sorted(zip(iterations, outputs), key=lambda pair: pair[0])
                    iterations, outputs = zip(*paired_sorted)

                    averages, lower, upper = zip(*outputs)
                    lower_errors = [avg - min_val for avg, min_val in zip(averages, lower)]
                    upper_errors = [max_val - avg for avg, max_val in zip(averages, upper)]

                    # Plot error bars separately
                    plt.errorbar(iterations, averages, yerr=[lower_errors, upper_errors], 
                                 fmt='none', ecolor='lightblue', elinewidth=1.5, capsize=3, label='_nolegend_')
                    # Plot data points separately with desired marker
                    plt.plot(iterations, averages, marker=ablation_markers[ablation], linestyle='-', color=ablation_colors[ablation], label=method, linewidth=1.5, markersize=6)

            # Plot baseline data
            for baseline_name in ['base', 'replay', 'EWC']:
                baseline_key = (baseline_name, cfg_common[1], cfg_common[3])
                if baseline_key in baselines:
                    baseline_avg, baseline_min, baseline_max = baselines[baseline_key]
                    baseline_avgs = [baseline_avg for _ in range(len(iterations))]
                    baseline_lower = baseline_avg - baseline_min
                    baseline_upper = baseline_max - baseline_avg
                    baseline_lowers = [baseline_lower for _ in range(len(iterations))]
                    baseline_uppers = [baseline_upper for _ in range(len(iterations))]
                    baseline_errors = [baseline_lowers, baseline_uppers]

                    if baseline_name == 'base':
                        baseline_label = 'Naive'
                        color_main = 'orange'
                        color_light = '#FFA07A'
                        marker = 's'
                    elif baseline_name == 'replay':
                        baseline_label = 'Replay'
                        color_main = 'green'
                        color_light = '#90EE90'
                        marker = '^'
                    elif baseline_name == 'EWC':
                        baseline_label = 'EWC'
                        color_main = 'red'
                        color_light = '#FF7F7F'
                        marker = 'D'

                    plt.errorbar(iterations, baseline_avgs, yerr=baseline_errors,
                                 fmt='none', ecolor=color_light, capsize=5, label='_nolegend_')
                    plt.plot(iterations, baseline_avgs, marker=marker, linestyle='-', color=color_main, markersize=8, label=f'{baseline_label}')

            plt.title(f"Affine Shift Plots for {strategy} - {cfg_common_str}", fontsize=16)
            plt.xlabel('Affine Shift Size', fontsize=14)
            plt.ylabel('Performance', fontsize=14)
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # Adjust layout to make room for the legend and minimize margins
            plt.tight_layout()

            # Place the legend outside the plot
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=True)

            # Save the plot to a page in the PDF
            pdf.savefig(bbox_inches='tight')
            plt.close()

    print(f"All affine shift plots saved to {pdf_path}")


def save_plot_to_pdf_(data, output_directory, name, cfg, baselines):
    # Unpack cfg into respective variables
    print(cfg, name)
    if name == "dim_size":
        met, mod, dat, aff, ite, part, infer = cfg
    elif name == "step_size":
        met, mod, ite, dat, aff, part, infer = cfg
    # Extract iterations and outputs from the list of tuples
    iterations = [int(item[0]) for item in data]
    outputs = [item[1] for item in data]

    # Sort based on iterations
    paired_sorted = sorted(zip(iterations, outputs), key=lambda pair: pair[0])
    iterations, outputs = zip(*paired_sorted)

    averages, lower, upper = zip(*outputs)
    lower_errors = [avg - min_val for avg, min_val in zip(averages, lower)]
    upper_errors = [max_val - avg for avg, max_val in zip(averages, upper)]

    # Plot main data with confidence intervals
    plt.figure()
    plt.errorbar(iterations, averages, yerr=[lower_errors, upper_errors], 
                 fmt='o-', ecolor='lightblue', elinewidth=3, capsize=0, linestyle='solid', label='SVD')

    # For baseline error bars: plot each baseline as a horizontal line with error bars
    for baseline_name in ['base', 'replay', 'EWC']:
        baseline_key = (baseline_name, mod, dat)
        if baseline_key in baselines:
            # Extract baseline average, min, and max values
            baseline_avg, baseline_min, baseline_max = baselines[baseline_key]
            # Calculate errors for baseline
            baseline_avgs = [baseline_avg for i in range(len(iterations))]
            baseline_lower = baseline_avg - baseline_min
            baseline_upper = baseline_max - baseline_avg
            baseline_lowers = [baseline_lower for i in range(len(iterations))]
            baseline_uppers = [baseline_upper for i in range(len(iterations))]
            baseline_errors = [baseline_lowers, baseline_uppers]

            if baseline_name == 'base':
                colour = 'orange'
            elif baseline_name == 'replay':
                colour = 'green'
            elif baseline_name == 'EWC':
                colour = 'red'

            # Plot baseline as a point in the middle of the x-axis with horizontal error bars
            plt.errorbar(iterations, baseline_avgs, yerr=baseline_errors,
                         fmt='s', ecolor=colour, capsize=5, 
                         label=f'{baseline_name}', linestyle='solid', markersize=8)

    plt.title(f"{met}, {mod}, {dat}, {aff}, {ite}, {part}")
    plt.xlabel(name)
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)

    # Save the plot to a PDF
    cfg_name = '_'.join(str(item) for item in cfg)
    filename = f"{cfg_name}_{name}.pdf"
    path = os.path.join(output_directory, filename)
    with PdfPages(path) as pdf:
        pdf.savefig()
        plt.close()

    print(f"Plot saved to {path}")

def save_plot_to_pdf(data, output_directory, name, cfg):
    met, mod, dat, aff, ite, part, infer = cfg

    # Extract iterations and outputs from the list of tuples
    iterations = [int(item[0]) for item in data]
    outputs = [item[1] for item in data]

    # Pair elements and sort pairs based on elements from the first list
    paired_sorted = sorted(zip(iterations, outputs), key=lambda pair: pair[0])

    # Unzip the pairs back into two lists
    iterations, outputs = zip(*paired_sorted)

    # Convert the tuples back to lists, if necessary
    iterations = list(iterations)
    outputs = list(outputs)
    averages, lower, upper = zip(*outputs)
    lower = [a - l for a, l in zip(averages, lower)]
    upper = [u - a for a, u in zip(averages, upper)]
    # Create the plot
    plt.figure()
    plt.errorbar(iterations, averages, yerr=[lower, upper], 
                fmt='o', 
                ecolor='lightblue', 
                elinewidth=3, 
                capsize=0, 
                linestyle='solid',
                label='Average with CI')

    #plt.plot(iterations, outputs, marker='o')  # 'o' creates a circle marker for each point
    plt.title(cfg)
    plt.xlabel(name)
    plt.ylabel('Avg Performance')
    plt.grid(True)

    # Save the plot to a PDF
    cfg_name = '_'.join(str(item) for item in cfg)  # Append file extension if needed

    namefile = cfg_name + name + '.pdf'
    path = os.path.join(output_directory, namefile)
    with PdfPages(path) as pdf:
        pdf.savefig()
        plt.close()

    print("Plot saved to pdf")

def append_to_list_in_dict(dic, key, value):
    """
    Appends a value to a list within a dictionary for the specified key.
    If the key does not exist, a new list for that key is created.

    Parameters:
    - dic: The dictionary where the list is stored.
    - key: The key associated with the list.
    - value: The value to append to the list.
    """
    if key not in dic:
        # If the key is not present, initialize a new list
        dic[key] = []
    # Append the value to the list associated with the key
    dic[key].append(value)


def calculate_mean_and_ci(data, confidence=0.90):
    mean = np.mean(data)
    sem = stats.sem(data)
    n = len(data)
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    mean = math.trunc(mean * 1000) / 1000 if isinstance(mean, (float, int)) and not pd.isna(mean) else mean
    ci_lower = math.trunc(ci_lower * 1000) / 1000 if isinstance(ci_lower, (float, int)) and not pd.isna(ci_lower) else ci_lower
    ci_upper = math.trunc(ci_upper * 1000) / 1000 if isinstance(ci_upper, (float, int)) and not pd.isna(ci_upper) else ci_upper
    return mean, ci_lower, ci_upper

def calculate_mean_and_mae(data, confidence=0.90):
    mean = np.mean(data)
    sem = stats.sem(data)
    n = len(data)
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    mean = math.trunc(mean * 1000) / 1000 if isinstance(mean, (float, int)) and not pd.isna(mean) else mean
    margin_of_error = math.trunc(margin_of_error * 1000) / 1000 if isinstance(margin_of_error, (float, int)) and not pd.isna(margin_of_error) else margin_of_error
    return mean, margin_of_error

def calculate_sum_and_ci(data, confidence=0.90):
    mean = np.sum(data)
    sem = stats.sem(data)
    n = len(data)
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    mean = math.trunc(mean * 1000) / 1000 if isinstance(mean, (float, int)) and not pd.isna(mean) else mean
    ci_lower = math.trunc(ci_lower * 1000) / 1000 if isinstance(ci_lower, (float, int)) and not pd.isna(ci_lower) else ci_lower
    ci_upper = math.trunc(ci_upper * 1000) / 1000 if isinstance(ci_upper, (float, int)) and not pd.isna(ci_upper) else ci_upper
    return mean, ci_lower, ci_upper

def calculate_metrics(table):
    """
    Calculate average performance at the last iteration and Backward Transfer (BWT).
    
    :param table: A DataFrame representing an upper triangular matrix with performance metrics. 
                  Columns and rows represent iterations.
    :return: The input DataFrame with added rows for 'Avg Performance' and 'BWT'.
    """
    last_col_index = table.columns[-1]  # Assumes the last column is where the final performance is stored
    
    # Check if the last column exists in the DataFrame
    if last_col_index not in table.columns:
        print(f"Missing required last column (index: {last_col_index}). Found columns: {table.columns}")
        return table

    # Calculate average performance at the last iteration for all Y
    last_col = table.loc[:last_col_index, last_col_index]
    performances = []
    #print(last_col)
    performances = [meanval for item in last_col if item is not np.nan for (meanval, _, _) in [item]]
    avg_performance = np.array(performances)
    avg_performance = calculate_mean_and_ci(avg_performance)
    table.loc['Avg Performance', last_col_index] = avg_performance

    # Calculate BWT (compare the diagonal elements and the last column)
    diagonal_values = np.diag(table)[:-1]
    diagonal_values = np.array([x[0] if isinstance(x, tuple) else x for x in diagonal_values])
    #print(diagonal_values)
    last_col_values = table.loc[:last_col_index-1, last_col_index]
    
    last_col_values = [x[0] if isinstance(x, tuple) else x for x in last_col_values]
    #print(last_col_values)
    #print(last_col_values)
    #sys.exit()

    bwt_values = (last_col_values - diagonal_values)
    bwt_mean, bwt_ci_lower, bwt_ci_upper = calculate_mean_and_ci(bwt_values)
    #bwt_mean = math.trunc(bwt_mean * 1000) / 1000 if isinstance(bwt_mean, (float, int)) and not pd.isna(bwt_mean) else bwt_mean
    #bwt_ci_lower = math.trunc(bwt_ci_lower * 1000) / 1000 if isinstance(bwt_ci_lower, (float, int)) and not pd.isna(bwt_ci_lower) else bwt_ci_lower
    #bwt_ci_upper = math.trunc(bwt_ci_upper * 1000) / 1000 if isinstance(bwt_ci_upper, (float, int)) and not pd.isna(bwt_ci_upper) else bwt_ci_upper

    table.loc['BWT', last_col_index] = (bwt_mean, bwt_ci_lower, bwt_ci_upper)

    #bwt = math.trunc(bwt * 1000) / 1000 if isinstance(bwt, (float, int)) and not pd.isna(bwt) else bwt
    #table.loc['BWT', last_col_index] = bwt
    
    return table


def format_cells(table_plot, table):
    #table_plot.scale(1.5,2)
    # Format the top row (column headers)
    for (i, j), cell in table_plot.get_celld().items():
        if i == 0 and j >= 0:  # Top row
            cell.set_facecolor('lightgrey')
            cell.set_text_props(fontweight='bold', ha='center', va='center')
            cell.set_edgecolor('black')

    # Format the leftmost column (row labels or index)
    for (i, j), cell in table_plot.get_celld().items():
        if j == -1 and i > 0:  # Leftmost column, excluding the (0, -1) cell
            cell.set_facecolor('lightgrey')
            cell.set_text_props(fontweight='bold', ha='center', va='center')
            cell.set_edgecolor('black')
    
    # Identify and format special rows ('Avg Performance' and 'BWT')
    num_rows = len(table.index)
    for (i, j), cell in table_plot.get_celld().items():
        if i > num_rows - 2:  # 'Avg Performance' and 'BWT' are the last two rows
            cell.set_text_props(ha='center', va='center')  # Align text to the right
            cell.set_facecolor('lightblue')  # Set background color (optional)


# Define your directory path here
directory_path = '/mnt/data4/joon/outputs/cl'
CL_method = 'svd_replay'
# Lists of expected strings for each part of the directory structure
expected_values = {
    'cl_method': ['svd_replay', 'PCA', 'svd_replay_alignment', 'svd_alignment'],
    'model': ['opt-1.3b', 'opt-350m', 'opt-125m', 'opt-2.7b', 'bloom-560m', 'bloom-1b1', 'phi-1_5', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
    'd_size': ['10', '32', '64', '100', '128','1024'],
    'dataset': ['Benchmark_500'],
    'seed': ['1234', '1331', '42'],
    'proj_method': ['affine'],
    'step_size': ['0', '1', '5', '10', '15', '20', '25', '50'],
    'partition': ['full', 'mha_only', 'ffn_only', 'qk_only', 'ov_only'],
    'infer': ['base', 'SVD', 'SVDreal', 'traintestbase', 'traintestSVD'],
}

# Dictionary to hold your data, structured as a search table
search_table = {}

# Regular expression to match the filenames
filename_pattern = re.compile(r'^results-([0-7])-([0-7])-(.+)\.json$')

# Mapping from file names to the metrics of interest
metric_mapping = {
    'C-STANCE': 'accuracy',
    'FOMC': 'accuracy',
    'ScienceQA': 'accuracy',
    'NumGLUE-cm': 'accuracy',
    'NumGLUE-ds': 'accuracy',
    'MeetingBank': 'rouge-L',
    'Py150': 'similarity',
    '20Minuten': 'sari'
}


# Adjusting the insertion of metric values into the search_table
for root, dirs, files in os.walk(directory_path):
    dirs[:] = [d for d in dirs if d in sum(expected_values.values(), [])]  # Filter out unexpected directories
    for file in files:
        filename_match = filename_pattern.match(file)
        if filename_match:
            x, y, file_name = filename_match.groups()

            # Construct the full file path
            file_path = os.path.join(root, file)

            # Validate and extract components from the directory path
            path_components = root.replace(directory_path, '').strip(os.sep).split(os.sep)

            if len(path_components) < 9 or not all(pc in expected_values[key] for pc, key in zip(path_components[:8], expected_values)):
                continue
            
            with open(file_path, 'r') as f:
                try:
                    json_data = json.load(f)
                except Exception as e:
                    print(e)
                    print(file_path)
                    sys.exit()


            eval_content = json_data.get('eval')
            if eval_content is None or file_name not in metric_mapping:
                continue
            
            metric = metric_mapping[file_name]
            metric_value = eval_content.get(metric)
            if metric_value is None:
                continue
            if file_name == '20Minuten' and 'sari' in metric_value[0].keys():
                metric_value = metric_value[0]['sari'] / 100.0
            if file_name == 'Py150':
                metric_value = metric_value / 100.0

            print(path_components)
            # Adjusting key to exclude the seed and task name for aggregation
            config_key = (
                path_components[0],  # cl_method
                path_components[1],  # model
                path_components[2],  # d_size
                path_components[3],  # dataset
                path_components[5],  # proj_method
                path_components[6],  # step_size
                path_components[7],  # partition
                path_components[8],  
                x, y, file_name
            )

            # Initialize the list for this configuration if it doesn't exist
            if config_key not in search_table:
                search_table[config_key] = []
            
            # Append the metric value to the list for this configuration
            search_table[config_key].append(metric_value)


# Example usage for a specific configuration
#config_key_example = ('PCA', 'opt-1.3b', '100', 'Benchmark_500', 'affine', '0', 'ffn_only', '7', '0', 'C-STANCE')
#print(search_table[config_key_example])

# Convert search table to tables
tables = {}
task_order = ['C-STANCE', 'FOMC', 'MeetingBank', 'Py150', 'ScienceQA', 'NumGLUE-cm', 'NumGLUE-ds', '20Minuten']

index_mapping = {
    '0' : 'C-STANCE',
    '1' : 'FOMC',
    '2' : 'MeetingBank',
    '3' : 'Py150',
    '4' : 'ScienceQA',
    '5' : 'NumGLUE-cm',
    '6' : 'NumGLUE-ds',
    '7' : '20Minuten'
}




for key, metric_values in search_table.items():
    config_key = key[:-3]
    x, y, name = key[-3:]

    if config_key not in tables:
        tables[config_key] = pd.DataFrame(index=range(8), columns=range(8))
    
    if name in task_order:
        tables[config_key].loc[int(y), int(x)] = calculate_mean_and_ci(metric_values)    


averages = {}
step_size_averages = {}
d_size_averages = {}

output_directory = '/mnt/data1/joon/outputs/tables'
save_performance_and_bwt_to_csv(tables, output_directory, 'clap')
outputname = CL_method + '_ci_tables.pdf'
output_pdf_path = os.path.join(output_directory, outputname)

# Parameters for layout
tables_per_page = 2  # e.g., 4 tables per page
num_rows = 2  # Number of rows of tables per page
num_cols = 1  # Number of columns of tables per page

with PdfPages(output_pdf_path) as pdf:
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(22,16))
    ax_idx = 0
    for config_key, table in tables.items():
        table = calculate_metrics(table)
        ax = axes.flat[ax_idx]  # Get the current axes
        ax.axis('off')  # Don't show the axis
        ax.set_title(f"Configuration: {config_key}", fontsize=16)  # Increase pad to bring title closer

        averages[config_key] = table.loc['Avg Performance', table.columns[-1]]
        

        #table = table.map(lambda x: math.trunc(x * 1000) / 1000 if isinstance(x, (float, int)) and not pd.isna(x) else x)
    
        # Replace NaN values with a blank
        table = table.fillna('-')
        table = table.rename(index=index_mapping)
        # Draw the table if it's not empty
        if not table.empty:
            table_plot = ax.table(cellText=table.values, colLabels=table.columns, rowLabels=table.index, loc='center', cellLoc='center')
            table_plot.auto_set_font_size(False)
            table_plot.set_fontsize(16)  # Adjust font size if needed
            table_plot.scale(1, 3)  # Adjust table scale (1 in x-direction, 2 in y-direction)
            
            # Format cells (header, index, special rows)
            format_cells(table_plot, table)
        else:
            ax.text(0.5, 0.5, 'No data available for this configuration', 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    transform=ax.transAxes,
                    fontsize=16)
        
        ax_idx += 1  # Move to the next subplot
        
        # If we have added the maximum number of tables per page or if it's the last table, save the page
        if ax_idx == tables_per_page or (config_key == list(tables.keys())[-1]):
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.6, wspace=0.3)  # Adjust spacing
            plt.tight_layout(pad=1.0)  # Adjust tight_layout padding
            pdf.savefig(fig, bbox_inches='tight')  # Save the current page
            plt.close(fig)  # Close the figure after saving to PDF
            
            if config_key != list(tables.keys())[-1]:  # If it's not the last table, prepare a new page
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 16))  # Same figure size for new page
                ax_idx = 0
    
    print(f"All tables saved in a single PDF: {output_pdf_path}")
    
    for key, avg in averages.items():
        step_key = key[:5] + key[6:]
        d_key = key[:2] + key[3:]
        s_size = key[5]
        d_size = key[2]
        append_to_list_in_dict(step_size_averages, step_key, (s_size, avg))
        append_to_list_in_dict(d_size_averages, d_key, (d_size, avg))


# Lists of expected strings for each part of the directory structure
expected_values = {
    'cl_method': ['base', 'replay', 'EWC', 'LFPT5', 'L2P'],
    'model': ['opt-1.3b', 'opt-350m', 'opt-125m', 'opt-2.7b', 'bloom-560m', 'bloom-1b1', 'phi-1_5', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
    'dataset': ['Benchmark_500'],
    'seed': ['1234', '1331', '42'],
}

# Dictionary to hold your data, structured as a search table
search_table_base = {}

# Regular expression to match the filenames
filename_pattern = re.compile(r'^results-([0-7])-([0-7])-(.+)\.json$')

# Mapping from file names to the metrics of interest
metric_mapping = {
    'C-STANCE': 'accuracy',
    'FOMC': 'accuracy',
    'ScienceQA': 'accuracy',
    'NumGLUE-cm': 'accuracy',
    'NumGLUE-ds': 'accuracy',
    'MeetingBank': 'rouge-L',
    'Py150': 'similarity',
    '20Minuten': 'sari',
}

directory_path = '/mnt/data1/joon/outputs/cl'

# Adjusting the insertion of metric values into the search_table
for root, dirs, files in os.walk(directory_path):
    dirs[:] = [d for d in dirs if d in sum(expected_values.values(), [])]  # Filter out unexpected directories
    for file in files:
        filename_match = filename_pattern.match(file)
        if filename_match:
            x, y, file_name = filename_match.groups()

            # Construct the full file path
            file_path = os.path.join(root, file)

            # Validate and extract components from the directory path
            path_components = root.replace(directory_path, '').strip(os.sep).split(os.sep)

            if len(path_components) < 3 or not all(pc in expected_values[key] for pc, key in zip(path_components[:2], expected_values)):
                continue
            
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            eval_content = json_data.get('eval')
            if eval_content is None or file_name not in metric_mapping:
                continue
            
            metric = metric_mapping[file_name]
            metric_value = eval_content.get(metric)
            if metric_value is None:
                continue
            if file_name == '20Minuten' and 'sari' in metric_value[0].keys():
                metric_value = metric_value[0]['sari'] / 100.0
            if file_name == 'Py150':
                metric_value = metric_value / 100.0
            

            # Adjusting key to exclude the seed and task name for aggregation
            config_key = (
                path_components[0],  # cl_method
                path_components[1],  # model
                path_components[2],  # dataset
                x, y, file_name
            )

            # Initialize the list for this configuration if it doesn't exist
            if config_key not in search_table_base:
                search_table_base[config_key] = []
            
            # Append the metric value to the list for this configuration
            search_table_base[config_key].append(metric_value)




# Convert search table to tables
tables = {}
task_order = ['C-STANCE', 'FOMC', 'MeetingBank', 'Py150', 'ScienceQA', 'NumGLUE-cm', 'NumGLUE-ds', '20Minuten']

index_mapping = {
    '0' : 'C-STANCE',
    '1' : 'FOMC',
    '2' : 'MeetingBank',
    '3' : 'Py150',
    '4' : 'ScienceQA',
    '5' : 'NumGLUE-cm',
    '6' : 'NumGLUE-ds',
    '7' : '20Minuten'
}

for key, metric_values in search_table_base.items():
    config_key = key[:-3]
    x, y, name = key[-3:]

    if config_key not in tables:
        tables[config_key] = pd.DataFrame(index=range(8), columns=range(8))
    
    if name in task_order:
        tables[config_key].loc[int(y), int(x)] = calculate_mean_and_ci(metric_values)

baseline_averages = {}

output_directory = '/mnt/data1/joon/outputs/tables'

output_pdf_path = os.path.join(output_directory, 'ci_baseline_tables.pdf')

save_performance_and_bwt_to_csv(tables, output_directory, 'baseline')



# Parameters for layout
tables_per_page = 2  # e.g., 4 tables per page
num_rows = 2  # Number of rows of tables per page
num_cols = 1  # Number of columns of tables per page

with PdfPages(output_pdf_path) as pdf:
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(22,16))
    ax_idx = 0
    for config_key, table in tables.items():
        table = calculate_metrics(table)
        ax = axes.flat[ax_idx]  # Get the current axes
        ax.axis('off')  # Don't show the axis
        ax.set_title(f"Configuration: {config_key}", fontsize=16)  # Increase pad to bring title closer
        baseline_averages[config_key] = table.loc['Avg Performance', table.columns[-1]]

        #table = table.map(lambda x: math.trunc(x * 1000) / 1000 if isinstance(x, (float, int)) and not pd.isna(x) else x)
    
        # Replace NaN values with a blank
        table = table.fillna('-')
        table = table.rename(index=index_mapping)
        # Draw the table if it's not empty
        if not table.empty:
            table_plot = ax.table(cellText=table.values, colLabels=table.columns, rowLabels=table.index, loc='center', cellLoc='center')
            table_plot.auto_set_font_size(False)
            table_plot.set_fontsize(16)  # Adjust font size if needed
            table_plot.scale(1, 3)  # Adjust table scale (1 in x-direction, 2 in y-direction)
            
            # Format cells (header, index, special rows)
            format_cells(table_plot, table)
        else:
            ax.text(0.5, 0.5, 'No data available for this configuration', 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    transform=ax.transAxes,
                    fontsize=16)
        
        ax_idx += 1  # Move to the next subplot
        
        # If we have added the maximum number of tables per page or if it's the last table, save the page
        if ax_idx == tables_per_page or (config_key == list(tables.keys())[-1]):
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.6, wspace=0.3)  # Adjust spacing
            plt.tight_layout(pad=1.0)  # Adjust tight_layout padding
            pdf.savefig(fig, bbox_inches='tight')  # Save the current page
            plt.close(fig)  # Close the figure after saving to PDF
            
            if config_key != list(tables.keys())[-1]:  # If it's not the last table, prepare a new page
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 16))  # Same figure size for new page
                ax_idx = 0
    
    print(f"All baseline tables saved in a single PDF: {output_pdf_path}")
    
    data_traintestbase_dict = extract_step_size_data(step_size_averages, 'traintestbase')
    data_traintestSVD_dict = extract_step_size_data(step_size_averages, 'traintestSVD')

    for cfgbase, cfgsvd in zip(data_traintestbase_dict.keys(), data_traintestSVD_dict.keys()):
        data_traintestbase = data_traintestbase_dict.get(cfgbase, [])
        data_traintestSVD = data_traintestSVD_dict.get(cfgsvd, [])
        #print(cfgbase)
        #print(cfgsvd)
        #print(data_traintestbase)
        #print(data_traintestSVD)
        if data_traintestbase and data_traintestSVD:
            save_comparison_plot_to_pdf(data_traintestbase, data_traintestSVD, output_directory, cfgbase)
    

    for cfg, avgs in step_size_averages.items():
        save_plot_to_pdf_improved(avgs, output_directory, 'step_size', cfg, baseline_averages)

    #save_all_affine_shift_plots_to_pdf(step_size_averages, output_directory, baseline_averages)


    for cfg, avgs in d_size_averages.items():
        save_plot_to_pdf_improved(avgs, output_directory, 'dim_size', cfg, baseline_averages)