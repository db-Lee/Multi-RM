import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import re
import argparse    

def get_methods_from_csv(df):
    """Extract and categorize methods from CSV, return selected methods and labels"""
    all_methods = df['Method'].unique()
    
    # Filter out Pass@N methods from the regular methods list
    regular_methods = [m for m in all_methods if not m.startswith('Pass@')]
    
    # Plot all regular methods
    methods_to_plot = regular_methods
    
    # Create labels for the selected methods
    method_labels = []
    for m in methods_to_plot:
        if m == 'MV':
            method_labels.append('Majority Voting')
        elif 'dORM' in m:
            method_labels.append('dORM')
        elif 'dPRM' in m:
            method_labels.append('dPRM')
        elif 'gORM' in m:
            method_labels.append('gORM')
        elif 'gPRM' in m:
            method_labels.append('gPRM')
        else:
            method_labels.append(m)
    
    print(f"All regular methods in CSV: {list(regular_methods)}")
    print(f"Methods to plot: {list(methods_to_plot)}")
    print(f"Labels: {method_labels}")
    
    return methods_to_plot, method_labels

def save_legend(method_labels, output_filename, plot_oracle):
    """Save legend as separate PDF and PNG files"""
    # Same colors, line styles, and markers as the main line plot
    colors = ['black', '#6699DD', '#FF7F7F', '#4472C4', '#E15759', 'gray']
    line_styles = ['-', '-', '-', '-', '-', ':']
    markers = ['o', 's', 'D', '^', 'v', 'x']
    
    # Create legend elements for the methods being plotted
    legend_elements = []
    
    num_methods = len(method_labels)
    for i in range(num_methods):
        legend_elements.append(Line2D([0], [0], 
                                     color=colors[i], 
                                     linestyle=line_styles[i],
                                     linewidth=2,
                                     marker=markers[i],
                                     markersize=10,
                                     alpha=0.9,
                                     label=method_labels[i]))
    
    # Add Pass@N only if plot_oracle is True
    if plot_oracle:
        legend_elements.append(Line2D([0], [0], 
                                     color=colors[5], 
                                     linestyle=line_styles[5],
                                     linewidth=2,
                                     marker=markers[5],
                                     markersize=10,
                                     alpha=0.9,
                                     label='Pass@N'))
    
    # Create figure for legend only
    fig, ax = plt.subplots(figsize=(6, 0.001), dpi=300)
    ax.axis('off')  # Hide axes
    
    # Create the legend
    ncol = num_methods + (1 if plot_oracle else 0)
    legend = ax.legend(handles=legend_elements, loc='center', ncol=ncol, 
                      fontsize=14, frameon=True, fancybox=True, shadow=True, framealpha=0.9,
                      handlelength=2, handletextpad=0.5, columnspacing=1)
    
    # Adjust layout to center the legend
    plt.tight_layout()
    
    # Save the legend
    legend_pdf = f'{output_filename}_legend.pdf'
    legend_png = f'{output_filename}_legend.png'
    
    plt.savefig(legend_pdf, bbox_inches='tight', 
               facecolor='white', edgecolor='none', pad_inches=0.1)
    print(f"Saved legend PDF: {legend_pdf}")
    
    plt.savefig(legend_png, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', pad_inches=0.1)
    print(f"Saved legend PNG: {legend_png}")
    
    plt.close()

def save_plots(df, methods_to_plot, method_labels, output_filename, y_margin_bottom, y_margin_top, plot_oracle):
    """Generate and save main plots as PDF and PNG"""
    # Get unique N values (sorted)
    N_values = sorted(df['N'].unique())
    
    # Get all domain columns (exclude N and Method columns)
    domain_columns = [col for col in df.columns if col not in ['N', 'Method']]
    # Filter to only mean columns (exclude std columns)
    mean_columns = [col for col in domain_columns if col.endswith('_mean')]
    
    # Extract domain names and create display names
    domains = [col.replace('_mean', '') for col in mean_columns]
    domain_display_names = [d.replace('_', ' ').title() for d in domains]
    
    # Professional colors and line styles
    colors = ['black', '#6699DD', '#FF7F7F', '#4472C4', '#E15759', 'gray']
    line_styles = ['-', '-', '-', '-', '-', ':']
    line_widths = [2, 2, 2, 2, 2, 2]
    markers = ['o', 's', 'D', '^', 'v', 'x']
    marker_sizes = [6, 6, 6, 6, 6, 8]
    
    # Set up the figure
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 4.5), dpi=300)
    
    # Create 3x5 subplot grid
    num_rows = 3
    num_cols = 5
    
    for i, (domain, display_name) in enumerate(zip(mean_columns, domain_display_names)):
        row = i // num_cols
        col = i % num_cols
        
        ax = plt.subplot(num_rows, num_cols, i + 1)
        
        # Plot each method across all N values
        for method_idx, (method, label) in enumerate(zip(methods_to_plot, method_labels)):
            method_data = []
            for n in N_values:
                # Get the value for this method at this N
                row_data = df[(df['N'] == n) & (df['Method'] == method)]
                if not row_data.empty:
                    method_data.append(row_data[domain].values[0])
                else:
                    method_data.append(np.nan)
            
            ax.plot(N_values, method_data,
                   color=colors[method_idx],
                   linestyle=line_styles[method_idx],
                   linewidth=line_widths[method_idx],
                   marker=markers[method_idx],
                   markersize=marker_sizes[method_idx],
                   label=label,
                   alpha=0.9)
        
        # Plot Pass@N only if plot_oracle is True
        if plot_oracle:
            pass_at_n_data = []
            pass_at_n_values = []
            for n in N_values:
                # Look for rows where Method is literally "Pass@N" (could also be Pass@1, Pass@2, etc)
                row_data = df[(df['N'] == n) & (df['Method'].str.startswith('Pass@'))]
                if not row_data.empty:
                    pass_at_n_data.append(row_data[domain].values[0])
                    pass_at_n_values.append(n)
            
            if pass_at_n_data:
                ax.plot(pass_at_n_values, pass_at_n_data,
                       color=colors[5],
                       linestyle=line_styles[5],
                       linewidth=line_widths[5],
                       marker=markers[5],
                       markersize=marker_sizes[5],
                       label='Pass@N',
                       alpha=0.9)
        
        ax.set_title(display_name, fontsize=12, pad=3, 
            fontweight='bold' if display_name == "Overall" else None)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks(N_values)
        if row == num_rows - 1:  # Bottom row
            ax.set_xticklabels([str(x) for x in N_values])
        else:
            ax.set_xticklabels([])
        
        all_values = []
        for method in methods_to_plot:
            for n in N_values:
                row_data = df[(df['N'] == n) & (df['Method'] == method)]
                if not row_data.empty and not np.isnan(row_data[domain].values[0]):
                    all_values.append(row_data[domain].values[0])
        
        if plot_oracle:
            for n in N_values:
                row_data = df[(df['N'] == n) & (df['Method'].str.startswith('Pass@'))]
                if not row_data.empty and not np.isnan(row_data[domain].values[0]):
                    all_values.append(row_data[domain].values[0])
        
        y_min = min(all_values) * y_margin_bottom
        y_max = max(all_values) * y_margin_top
        ax.set_ylim(y_min, y_max)
        
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        if display_name == "Overall":
            for spine in ax.spines.values():
                spine.set_linewidth(2)
        
        ax.tick_params(axis='both', labelsize=10)
    
    fig.supxlabel('Number of CoTs (N)', fontsize=12, y=0.05)
    fig.supylabel('Task accuracy (%)', fontsize=12, x=0.02)
    
    plt.tight_layout(pad=0.9, h_pad=0.5, w_pad=0.5)
    
    pdf_filename = f'{output_filename}.pdf'
    png_filename = f'{output_filename}.png'
    
    plt.savefig(pdf_filename, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"Saved plot PDF: {pdf_filename}")
    
    plt.savefig(png_filename, bbox_inches='tight', 
               facecolor='white', edgecolor='none', dpi=300)
    print(f"Saved plot PNG: {png_filename}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate subplot visualizations from CSV data')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--plot_oracle', action='store_true')
    parser.add_argument('--y_margin_bottom', type=float, default=0.97)
    parser.add_argument('--y_margin_top', type=float, default=1.03)
    args = parser.parse_args()
    
    # Read the CSV file
    print(f"Reading CSV file: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    print(f"\nCSV shape: {df.shape}")
    print(f"Unique N values: {sorted(df['N'].unique())}")
    print(f"Unique methods: {df['Method'].unique()}")
    print(f"Plot oracle (Pass@N): {args.plot_oracle}")
    
    # Get methods to plot
    methods_to_plot, method_labels = get_methods_from_csv(df)
    print(f"\nPlotting {len(methods_to_plot)} methods: {method_labels}")
    if args.plot_oracle:
        print("Plus Pass@N oracle baseline")
    
    # Save main plots
    save_plots(df, methods_to_plot, method_labels, args.output_file, 
               args.y_margin_bottom, args.y_margin_top, args.plot_oracle)
    
    # Save legend separately
    save_legend(method_labels, args.output_file, args.plot_oracle)
    
    print(f"\nAll outputs saved with prefix: {args.output_file}")

if __name__ == '__main__':
    main()