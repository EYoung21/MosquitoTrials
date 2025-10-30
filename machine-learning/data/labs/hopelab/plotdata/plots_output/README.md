# EPG Data Analysis Report

**Date:** October 24, 2025  
**Data Source:** `/data/labs/hopelab/epg/tarsalis_data_clean/cxthand1sep2021no1.csv`  
**Analysis Script:** `plotSampleData.py`

## Dataset Overview

- **Total Duration:** 847.61 seconds (~14.1 minutes)
- **Sampling Rate:** ~100 Hz
- **Total Samples:** 84,762
- **Number of Probes:** 5
- **Unique Labels:** 6 (J, K, L, M, NP, Z)

## Label Distribution

The data contains the following label types with their proportions:

| Label | Count  | Proportion |
|-------|--------|------------|
| L     | 34,336 | 40.5%      |
| NP    | 27,559 | 32.5%      |
| M     | 18,055 | 21.3%      |
| J     | 3,160  | 3.7%       |
| K     | 1,552  | 1.8%       |
| Z     | 100    | 0.1%       |

**Notes:**
- **NP** (Non-Probing): Periods when the mosquito is not actively probing
- The most common active probing behavior is **L** (40.5%)
- **Z** labels appear very rarely (0.1%) and as very short sequences

## Probe Analysis

Probes are continuous sequences of non-NP activity:

- **Mean probe duration:** 114.40 seconds
- **Median probe duration:** 84.20 seconds
- **Range:** 68.09 - 248.14 seconds
- **Total probes found:** 5

## Generated Visualizations

### Summary and Overview
- **`00_summary.png`** - Comprehensive overview with label distribution, sample data, probe statistics, and recording info

### Label Analysis
- **`01_label_distribution.png`** - Bar charts showing label counts and proportions

### Data Snippets (10-second windows)
- **`02_snippet_1.png`** - Beginning of recording (0-10s)
- **`02_snippet_2.png`** - First quarter (212-222s)
- **`02_snippet_3.png`** - Middle (424-434s)
- **`02_snippet_4.png`** - Third quarter (636-646s)

### Rectification Comparison
- **`03_rectification_comparison.png`** - Side-by-side comparison of pre-rectified and post-rectified signals
  - Pre-rectified: Original signal with both positive and negative values
  - Post-rectified: Absolute value transformation, all positive

### Data with Label Overlays
- **`04_with_labels_1.png`** - 10-second snippet with colored regions showing different label types
- **`04_with_labels_2.png`** - Another snippet from middle of recording
- **`04_with_labels_3.png`** - Another snippet from later in recording

### Probe Visualizations
- **`05_probe_durations.png`** - Histogram of probe duration distribution
- **`06_probe_1.png`** through **`06_probe_5.png`** - Individual visualizations of each of the 5 probes with labels

### Label-Specific Examples
- **`07_label_J_examples.png`** - 4 examples of label J sequences (5 sequences found)
- **`07_label_K_examples.png`** - 4 examples of label K sequences (5 sequences found)
- **`07_label_L_examples.png`** - 4 examples of label L sequences (5 sequences found)
- **`07_label_M_examples.png`** - 1 example of label M sequence (only 1 long sequence found)

Note: Label Z had 0 sequences longer than 50 samples (0.5 seconds) and thus no examples were plotted.

## Key Observations

### Signal Characteristics

1. **Pre-rectified signal:** Shows oscillations around ~-8V baseline with variations
2. **Post-rectified signal:** Shows absolute values, making peaks more visible for analysis
3. **Sampling rate:** ~100 Hz provides good temporal resolution for detecting feeding behaviors

### Label Patterns

1. **NP (Non-Probing):** 
   - Very stable signal near baseline
   - Low amplitude variations
   - Represents resting or searching behavior

2. **L (Most common - 40.5%):**
   - Highly variable patterns
   - Can show both large amplitude spikes and quieter periods
   - Often forms long continuous sequences

3. **M (Second most common - 21.3%):**
   - Tends to form one long continuous sequence
   - Shows distinct waveform patterns

4. **J and K (Less common - 3.7% and 1.8%):**
   - Shorter sequences
   - Distinct waveform characteristics
   - Often appear as transitions between other states

5. **Z (Rare - 0.1%):**
   - Very short sequences (all < 50 samples or 0.5s)
   - May represent brief transitional states or artifacts
   - Note: In model_eval.py, Z is replaced with W

### Probe Structure

- Probes range from ~1 to ~4 minutes in duration
- Most probes show transitions between multiple label types
- NP periods separate individual probes
- Probe durations show variability, suggesting different feeding strategies or host factors

## Analysis Methods

The analysis script (`plotSampleData.py`) includes:

1. **Data loading and exploration** - Using pandas to load and inspect the CSV data
2. **Label distribution analysis** - Counting and visualizing label frequencies
3. **Time series visualization** - Plotting voltage traces over time
4. **Rectification comparison** - Comparing pre- and post-rectified signals
5. **Label overlay visualization** - Color-coded regions showing label assignments
6. **Probe finding algorithm** - Identifying continuous non-NP sequences (from `model_eval.py`)
7. **Label-specific sequence extraction** - Finding and visualizing examples of each label type

## Usage

To regenerate these plots:

```bash
cd /Users/eliyoung/hmc-epg-project/machine-learning/data/labs/hopelab/plotdata
source venv/bin/activate
python plotSampleData.py
```

The script will create a `plots_output/` directory and save all visualizations there.

