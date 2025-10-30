import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob

# Close any existing plots
plt.close("all")

# Set random seed for reproducibility
np.random.seed(123456)

# Create output directory for plots
output_dir = "plots_output"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# PART 1: Load the data
# ============================================================================
print("Loading data...")
data_path = "/data/labs/hopelab/epg/tarsalis_data_clean"
filenames = glob.glob(os.path.expanduser(f"{data_path}/*.csv"))
print(f"Found {len(filenames)} data files:")
for fname in filenames:
    print(f"  - {fname}")

# Load the first file (or you can loop through all files)
df = pd.read_csv(filenames[0])
print(f"\nData shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# PART 2: Analyze label distribution
# ============================================================================
print("\n" + "="*60)
print("LABEL DISTRIBUTION ANALYSIS")
print("="*60)
label_counts = df['labels'].value_counts()
label_proportions = df['labels'].value_counts(normalize=True)

print("\nLabel counts:")
print(label_counts)
print("\nLabel proportions:")
print(label_proportions)

# Create a bar plot of label distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
label_counts.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Label Counts')
ax1.set_xlabel('Label')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=0)

label_proportions.plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('Label Proportions')
ax2.set_xlabel('Label')
ax2.set_ylabel('Proportion')
ax2.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f"{output_dir}/01_label_distribution.png", dpi=150)
print(f"Saved: {output_dir}/01_label_distribution.png")

# ============================================================================
# PART 3: Plot snippets of data (5-10 seconds)
# ============================================================================
print("\n" + "="*60)
print("PLOTTING DATA SNIPPETS")
print("="*60)

def plot_snippet(df, start_time, duration, title, filename):
    """
    Plot a snippet of the data
    
    Args:
        df: DataFrame containing the data
        start_time: Start time in seconds
        duration: Duration in seconds
        title: Title for the plot
        filename: Filename to save the plot
    """
    # Get the snippet
    mask = (df['time'] >= start_time) & (df['time'] < start_time + duration)
    snippet = df[mask]
    
    if len(snippet) == 0:
        print(f"No data found for time range {start_time}-{start_time+duration}s")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(snippet['time'], snippet['pre_rect'], color='black', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (pre-rectified)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# Plot several 10-second snippets from different parts of the recording
max_time = df['time'].max()
snippet_times = [0, max_time * 0.25, max_time * 0.5, max_time * 0.75]

for i, start_t in enumerate(snippet_times):
    plot_snippet(df, start_t, 10, 
                 f"Data Snippet {i+1}: {start_t:.1f}-{start_t+10:.1f}s",
                 f"{output_dir}/02_snippet_{i+1}.png")

# ============================================================================
# PART 4: Compare pre-rect vs post-rect
# ============================================================================
print("\n" + "="*60)
print("COMPARING PRE-RECT VS POST-RECT")
print("="*60)

def plot_rectification_comparison(df, start_time, duration, filename):
    """
    Plot pre-rectified and post-rectified data side by side
    """
    mask = (df['time'] >= start_time) & (df['time'] < start_time + duration)
    snippet = df[mask]
    
    if len(snippet) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Pre-rectified
    ax1.plot(snippet['time'], snippet['pre_rect'], color='blue', linewidth=0.5)
    ax1.set_ylabel('Voltage (pre-rectified)')
    ax1.set_title(f'Pre-Rectified Signal ({start_time:.1f}-{start_time+duration:.1f}s)')
    ax1.grid(True, alpha=0.3)
    
    # Post-rectified
    ax2.plot(snippet['time'], snippet['post_rect'], color='red', linewidth=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (post-rectified)')
    ax2.set_title(f'Post-Rectified Signal ({start_time:.1f}-{start_time+duration:.1f}s)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# Compare rectification for a 10-second snippet
plot_rectification_comparison(df, max_time * 0.3, 10, 
                               f"{output_dir}/03_rectification_comparison.png")

# ============================================================================
# PART 5: Visualize data with labels (borrowed from model_eval.py)
# ============================================================================
print("\n" + "="*60)
print("VISUALIZING DATA WITH LABELS")
print("="*60)

def plot_with_labels(time, voltage, labels, title, filename):
    """
    Plot voltage data with colored regions indicating labels
    (Based on plot_labels function from model_eval.py)
    """
    label_to_color = {
        "NP": "red",
        "J": "blue",
        "K": "green",
        "L": "purple",
        "M": "pink",
        "N": "cyan",
        "W": "orange"
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the voltage trace
    ax.plot(time, voltage, color='black', linewidth=0.5, zorder=1)
    
    # Get the y-axis limits for filling
    fill_min, fill_max = voltage.min(), voltage.max()
    
    # Fill regions for each label
    for label, color in label_to_color.items():
        if label in labels.values:
            fill = ax.fill_between(time, fill_min, fill_max,
                                   where=(labels == label), 
                                   color=color, alpha=0.3, zorder=0)
            fill.set_label(label)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (pre-rectified)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Plot several snippets with labels
for i, start_t in enumerate([0, max_time * 0.3, max_time * 0.6]):
    mask = (df['time'] >= start_t) & (df['time'] < start_t + 10)
    snippet = df[mask]
    
    if len(snippet) > 0:
        plot_with_labels(snippet['time'], snippet['pre_rect'], snippet['labels'],
                        f"Data with Labels: {start_t:.1f}-{start_t+10:.1f}s",
                        f"{output_dir}/04_with_labels_{i+1}.png")

# ============================================================================
# PART 6: Probe finding and label separation
# ============================================================================
print("\n" + "="*60)
print("SEPARATING PROBES AND ANALYZING LABEL PATTERNS")
print("="*60)

def leak_probe_finder(labels):
    """
    Find probe indices (sequences without NP)
    (Borrowed from DataImport class in model_eval.py)
    """
    non_np_indices = np.where(labels != 'NP')[0]
    probes = []
    start = 0
    end = 0
    for i in range(len(non_np_indices)):
        if i == 0:
            start = non_np_indices[i]
        elif i == len(non_np_indices) - 1:
            end = non_np_indices[i]
        elif abs(non_np_indices[i] - non_np_indices[i - 1]) > 1:
            start = non_np_indices[i]
        elif abs(non_np_indices[i] - non_np_indices[i + 1]) > 1:
            end = non_np_indices[i]
        if start > 0 and end > 0:
            probes.append((start, end))
            start = 0
            end = 0
    return probes

# Find probes in the data
probe_indices = leak_probe_finder(df["labels"].values)
print(f"\nFound {len(probe_indices)} probes")

# Extract probes as separate dataframes
probes = [df.iloc[start:end+1].reset_index(drop=True).copy() 
          for start, end in probe_indices]

# Analyze probe durations
probe_durations = [(probe['time'].max() - probe['time'].min()) for probe in probes]
print(f"\nProbe duration statistics:")
print(f"  Mean: {np.mean(probe_durations):.2f}s")
print(f"  Median: {np.median(probe_durations):.2f}s")
print(f"  Min: {np.min(probe_durations):.2f}s")
print(f"  Max: {np.max(probe_durations):.2f}s")

# Plot histogram of probe durations
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(probe_durations, bins=30, color='steelblue', edgecolor='black')
ax.set_xlabel('Probe Duration (s)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Probe Durations')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/05_probe_durations.png", dpi=150)
plt.close()
print(f"Saved: {output_dir}/05_probe_durations.png")

# Plot first few probes
for i, probe in enumerate(probes[:5]):  # Plot first 5 probes
    if len(probe) > 0:
        plot_with_labels(probe['time'], probe['pre_rect'], probe['labels'],
                        f"Probe {i+1} (Duration: {probe['time'].max() - probe['time'].min():.2f}s)",
                        f"{output_dir}/06_probe_{i+1}.png")

# ============================================================================
# PART 7: Analyze and plot examples of each label type
# ============================================================================
print("\n" + "="*60)
print("EXTRACTING EXAMPLES OF EACH LABEL TYPE")
print("="*60)

def find_label_sequences(df, target_label, min_length=100):
    """
    Find continuous sequences of a specific label
    """
    labels = df['labels'].values
    sequences = []
    
    in_sequence = False
    start_idx = 0
    
    for i, label in enumerate(labels):
        if label == target_label and not in_sequence:
            # Start of a new sequence
            in_sequence = True
            start_idx = i
        elif label != target_label and in_sequence:
            # End of sequence
            if i - start_idx >= min_length:
                sequences.append((start_idx, i))
            in_sequence = False
    
    # Check if we ended in a sequence
    if in_sequence and len(labels) - start_idx >= min_length:
        sequences.append((start_idx, len(labels)))
    
    return sequences

# Get all unique labels (excluding NP)
unique_labels = [l for l in df['labels'].unique() if l != 'NP']
print(f"\nUnique labels (excluding NP): {unique_labels}")

# For each label, find and plot examples
for label in unique_labels:
    sequences = find_label_sequences(df, label, min_length=50)
    print(f"\nLabel '{label}': Found {len(sequences)} sequences")
    
    if len(sequences) == 0:
        continue
    
    # Create a figure with multiple examples (up to 4)
    n_examples = min(4, len(sequences))
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3*n_examples))
    
    if n_examples == 1:
        axes = [axes]
    
    for i, (start, end) in enumerate(sequences[:n_examples]):
        # Get a window around the sequence (with some context before/after)
        context = 200  # samples
        window_start = max(0, start - context)
        window_end = min(len(df), end + context)
        
        snippet = df.iloc[window_start:window_end]
        
        axes[i].plot(snippet['time'], snippet['pre_rect'], 
                    color='black', linewidth=0.5)
        
        # Highlight the specific label region
        label_snippet = df.iloc[start:end]
        fill_min, fill_max = snippet['pre_rect'].min(), snippet['pre_rect'].max()
        axes[i].fill_between(label_snippet['time'], fill_min, fill_max,
                            alpha=0.3, color='yellow', label=f'Label {label}')
        
        axes[i].set_ylabel('Voltage')
        axes[i].set_title(f"Label '{label}' Example {i+1} "
                         f"(Duration: {label_snippet['time'].max() - label_snippet['time'].min():.2f}s)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_label_{label}_examples.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/07_label_{label}_examples.png")

# ============================================================================
# PART 8: Create a comprehensive summary report
# ============================================================================
print("\n" + "="*60)
print("CREATING SUMMARY VISUALIZATION")
print("="*60)

# Create a comprehensive summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Label distribution bar chart
ax1 = fig.add_subplot(gs[0, 0])
label_counts.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Label Distribution (Counts)', fontweight='bold')
ax1.set_xlabel('Label')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=0)

# 2. Label proportions pie chart
ax2 = fig.add_subplot(gs[0, 1])
label_proportions.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90)
ax2.set_title('Label Proportions', fontweight='bold')
ax2.set_ylabel('')

# 3. Sample data with labels
ax3 = fig.add_subplot(gs[1, :])
sample_start = max_time * 0.4
sample_duration = 20
mask = (df['time'] >= sample_start) & (df['time'] < sample_start + sample_duration)
sample = df[mask]

if len(sample) > 0:
    ax3.plot(sample['time'], sample['pre_rect'], color='black', linewidth=0.5)
    fill_min, fill_max = sample['pre_rect'].min(), sample['pre_rect'].max()
    
    label_to_color = {
        "NP": "red", "J": "blue", "K": "green", 
        "L": "purple", "M": "pink", "N": "cyan", "W": "orange"
    }
    
    for label, color in label_to_color.items():
        if label in sample['labels'].values:
            ax3.fill_between(sample['time'], fill_min, fill_max,
                           where=(sample['labels'] == label),
                           color=color, alpha=0.3, label=label)
    
    ax3.legend(loc='upper right', ncol=7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Voltage (pre-rectified)')
    ax3.set_title(f'Sample Recording with Labels ({sample_start:.1f}-{sample_start+sample_duration:.1f}s)', 
                  fontweight='bold')
    ax3.grid(True, alpha=0.3)

# 4. Probe duration distribution
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(probe_durations, bins=20, color='coral', edgecolor='black')
ax4.set_xlabel('Probe Duration (s)')
ax4.set_ylabel('Count')
ax4.set_title('Probe Duration Distribution', fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Data statistics table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

stats_data = [
    ['Total Duration', f'{df["time"].max():.2f} s'],
    ['Sampling Rate', f'{1/(df["time"].iloc[1] - df["time"].iloc[0]):.2f} Hz'],
    ['Total Samples', f'{len(df):,}'],
    ['Number of Probes', f'{len(probes)}'],
    ['Mean Probe Duration', f'{np.mean(probe_durations):.2f} s'],
    ['Unique Labels', f'{len(df["labels"].unique())}'],
]

table = ax5.table(cellText=stats_data, cellLoc='left',
                 colWidths=[0.6, 0.4],
                 loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the table
for i in range(len(stats_data)):
    cell = table[(i, 0)]
    cell.set_facecolor('#E8E8E8')
    cell.set_text_props(weight='bold')

ax5.set_title('Recording Statistics', fontweight='bold', pad=20)

plt.suptitle('EPG Data Analysis Summary', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f"{output_dir}/00_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/00_summary.png")

# ============================================================================
# DONE!
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\nAll plots have been saved to: {output_dir}/")
print("\nGenerated files:")
for fname in sorted(os.listdir(output_dir)):
    print(f"  - {fname}")
