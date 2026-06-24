"""
analysis.py

Quantitative analysis for the climate thinking partner study.
Produces:
  - analysis_clean.csv        : cleaned data with composite scores
  - results_descriptives.csv  : means, SDs, medians per condition
  - results_assumptions.csv   : Shapiro-Wilk and Levene test results
  - results_anova.csv         : one-way ANOVA results per outcome
  - results_posthoc.csv       : Tukey HSD post-hoc comparisons
  - results_kruskal.csv       : Kruskal-Wallis (non-parametric backup)
  - figures/                  : box plots and distribution plots

Usage:
    python3 analysis.py
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV   = 'survey_scores.csv'   # adjust path if needed
OUTPUT_DIR  = '.'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

CONDITION_LABELS = {
    'climate_thinking_partner_steelman': 'Steelman',
    'climate_thinking_partner_socratic':  'Socratic',
    'climate_thinking_partner_neutral':   'Neutral',
}
CONDITION_ORDER  = ['Steelman', 'Socratic', 'Neutral']
CONDITION_COLORS = ['#6B3FA0', '#2E86AB', '#A23B72']

TRUST_COLS    = ['partner_trust_1_to_10', 'partner_well_reasoned_1_to_5', 'partner_honest_1_to_5']
AUTONOMY_COLS = ['perceived_autonomy_1_to_7', 'augmentation_vs_replacement_1_to_5', 'independent_capability_1_to_5']
QUANT_REQUIRED = [
    'pre_opinion_1_to_7', 'pre_opinion_confidence_1_to_7',
    'post_opinion_1_to_7', 'post_opinion_confidence_1_to_7',
    'reconsideration_1_to_7',
] + TRUST_COLS + AUTONOMY_COLS

def sep(title=''):
    print('\n' + '=' * 65)
    if title:
        print(f'  {title}')
        print('=' * 65)


# ── 1. Load and clean ─────────────────────────────────────────────────────────

sep('1. LOADING AND CLEANING DATA')

df_raw = pd.read_csv(INPUT_CSV)
print(f'Raw rows: {len(df_raw)}')

# Drop rows missing any required quantitative field
df = df_raw.dropna(subset=QUANT_REQUIRED).copy()
print(f'After dropping incomplete rows: {len(df)} (removed {len(df_raw) - len(df)})')

# Map condition names to short labels
df['condition_label'] = df['condition'].map(CONDITION_LABELS)
print('\nPer condition after cleaning:')
print(df['condition_label'].value_counts().to_string())


# ── 2. Build composite scores ─────────────────────────────────────────────────

sep('2. COMPOSITE SCORES')

# Rescale partner_trust to 1-5 (divide by 2) then average with other trust items
df['trust_rescaled']    = df['partner_trust_1_to_10'] / 2
df['trust_composite']   = df[['trust_rescaled', 'partner_well_reasoned_1_to_5',
                               'partner_honest_1_to_5']].mean(axis=1)

# Rescale perceived_autonomy to 1-5 (divide by 1.4) then average with other autonomy items
df['autonomy_rescaled'] = df['perceived_autonomy_1_to_7'] / 1.4
df['autonomy_composite'] = df[['autonomy_rescaled', 'augmentation_vs_replacement_1_to_5',
                                'independent_capability_1_to_5']].mean(axis=1)

# Opinion change score already in CSV; recalculate cleanly
df['opinion_change']    = df['post_opinion_1_to_7'] - df['pre_opinion_1_to_7']
df['confidence_change'] = df['post_opinion_confidence_1_to_7'] - df['pre_opinion_confidence_1_to_7']

OUTCOMES = {
    'opinion_change':    'Opinion Change (SQ1)',
    'trust_composite':   'Epistemic Trust (SQ2)',
    'autonomy_composite':'Epistemic Autonomy (SQ3)',
}

# Cronbach's alpha helper
def cronbach_alpha(df_items):
    k = df_items.shape[1]
    item_vars = df_items.var(ddof=1, axis=0).sum()
    total_var = df_items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars / total_var)

alpha_trust    = cronbach_alpha(df[['trust_rescaled', 'partner_well_reasoned_1_to_5', 'partner_honest_1_to_5']])
alpha_autonomy = cronbach_alpha(df[['autonomy_rescaled', 'augmentation_vs_replacement_1_to_5', 'independent_capability_1_to_5']])
print(f"Cronbach's alpha — Trust composite:    {alpha_trust:.3f}")
print(f"Cronbach's alpha — Autonomy composite: {alpha_autonomy:.3f}")
print("(Acceptable if α > 0.60)")

# Save clean CSV
clean_cols = ['session_id', 'condition', 'condition_label', 'persona_index',
              'pre_opinion_1_to_7', 'post_opinion_1_to_7',
              'opinion_change', 'confidence_change',
              'trust_composite', 'autonomy_composite',
              'pre_opinion_confidence_1_to_7', 'post_opinion_confidence_1_to_7',
              'reconsideration_1_to_7']
df[clean_cols].to_csv(os.path.join(OUTPUT_DIR, 'analysis_clean.csv'), index=False)
print('\nSaved: analysis_clean.csv')


# ── 3. Descriptive statistics ─────────────────────────────────────────────────

sep('3. DESCRIPTIVE STATISTICS')

desc_rows = []
for outcome_col, outcome_label in OUTCOMES.items():
    for cond in CONDITION_ORDER:
        vals = df[df['condition_label'] == cond][outcome_col].dropna()
        desc_rows.append({
            'Outcome':   outcome_label,
            'Condition': cond,
            'N':         len(vals),
            'Mean':      round(vals.mean(), 3),
            'SD':        round(vals.std(ddof=1), 3),
            'Median':    round(vals.median(), 3),
            'Min':       round(vals.min(), 3),
            'Max':       round(vals.max(), 3),
        })

desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))
desc_df.to_csv(os.path.join(OUTPUT_DIR, 'results_descriptives.csv'), index=False)
print('\nSaved: results_descriptives.csv')


# ── 4. Assumption checks ──────────────────────────────────────────────────────

sep('4. ASSUMPTION CHECKS')

assump_rows = []

for outcome_col, outcome_label in OUTCOMES.items():
    groups = [df[df['condition_label'] == c][outcome_col].dropna().values
              for c in CONDITION_ORDER]

    # Shapiro-Wilk per condition
    for i, cond in enumerate(CONDITION_ORDER):
        if len(groups[i]) >= 3:
            stat, p = stats.shapiro(groups[i])
            normal = 'YES' if p > 0.05 else 'NO'
            assump_rows.append({
                'Outcome': outcome_label, 'Test': 'Shapiro-Wilk',
                'Condition': cond, 'Statistic': round(stat, 4),
                'p-value': round(p, 4), 'Assumption met': normal
            })

    # Levene's test (homogeneity of variance)
    stat, p = stats.levene(*groups)
    homogeneous = 'YES' if p > 0.05 else 'NO'
    assump_rows.append({
        'Outcome': outcome_label, 'Test': "Levene's",
        'Condition': 'All', 'Statistic': round(stat, 4),
        'p-value': round(p, 4), 'Assumption met': homogeneous
    })

assump_df = pd.DataFrame(assump_rows)
print(assump_df.to_string(index=False))
assump_df.to_csv(os.path.join(OUTPUT_DIR, 'results_assumptions.csv'), index=False)
print('\nSaved: results_assumptions.csv')


# ── 5. One-way ANOVA ──────────────────────────────────────────────────────────

sep('5. ONE-WAY ANOVA')

anova_rows = []

for outcome_col, outcome_label in OUTCOMES.items():
    groups = [df[df['condition_label'] == c][outcome_col].dropna().values
              for c in CONDITION_ORDER]
    n_total = sum(len(g) for g in groups)
    k = len(groups)

    f_stat, p_val = stats.f_oneway(*groups)

    # Eta-squared
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total   = sum(((v - grand_mean)**2) for g in groups for v in g)
    eta_sq     = ss_between / ss_total if ss_total > 0 else 0

    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))

    anova_rows.append({
        'Outcome':    outcome_label,
        'F(2,N-3)':  f'F(2, {n_total - k})',
        'F-stat':    round(f_stat, 4),
        'p-value':   round(p_val, 4),
        'eta-sq':    round(eta_sq, 4),
        'Sig':       sig,
    })
    print(f"{outcome_label}: F(2,{n_total-k})={f_stat:.4f}, p={p_val:.4f}, η²={eta_sq:.4f} {sig}")

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(os.path.join(OUTPUT_DIR, 'results_anova.csv'), index=False)
print('\nSaved: results_anova.csv')


# ── 6. Kruskal-Wallis (non-parametric backup) ─────────────────────────────────

sep('6. KRUSKAL-WALLIS (non-parametric backup)')

kruskal_rows = []
for outcome_col, outcome_label in OUTCOMES.items():
    groups = [df[df['condition_label'] == c][outcome_col].dropna().values
              for c in CONDITION_ORDER]
    h_stat, p_val = stats.kruskal(*groups)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    kruskal_rows.append({
        'Outcome':  outcome_label,
        'H-stat':   round(h_stat, 4),
        'p-value':  round(p_val, 4),
        'Sig':      sig,
    })
    print(f"{outcome_label}: H={h_stat:.4f}, p={p_val:.4f} {sig}")

kruskal_df = pd.DataFrame(kruskal_rows)
kruskal_df.to_csv(os.path.join(OUTPUT_DIR, 'results_kruskal.csv'), index=False)
print('\nSaved: results_kruskal.csv')


# ── 7. Tukey HSD post-hoc ────────────────────────────────────────────────────

sep('7. TUKEY HSD POST-HOC')

from scipy.stats import studentized_range

def tukey_hsd(groups, labels):
    """Manual Tukey HSD for three groups."""
    rows = []
    n_groups = len(groups)
    n_total  = sum(len(g) for g in groups)
    k        = n_groups
    ms_within = sum(((v - g.mean())**2) for g in groups for v in g) / (n_total - k)

    pairs = [(0,1),(0,2),(1,2)]
    for i, j in pairs:
        g1, g2 = groups[i], groups[j]
        mean_diff = g1.mean() - g2.mean()
        se = np.sqrt(ms_within / 2 * (1/len(g1) + 1/len(g2)))
        q  = abs(mean_diff) / se if se > 0 else 0
        # p-value from studentized range distribution
        p  = 1 - studentized_range.cdf(q, k, n_total - k)
        rows.append({
            'Group A': labels[i], 'Group B': labels[j],
            'Mean diff (A-B)': round(mean_diff, 4),
            'q-stat': round(q, 4),
            'p-adj':  round(p, 4),
            'Sig': '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        })
    return rows

posthoc_rows = []
for outcome_col, outcome_label in OUTCOMES.items():
    groups = [df[df['condition_label'] == c][outcome_col].dropna().values
              for c in CONDITION_ORDER]
    rows = tukey_hsd(groups, CONDITION_ORDER)
    for r in rows:
        r['Outcome'] = outcome_label
        posthoc_rows.append(r)
    print(f"\n{outcome_label}:")
    for r in rows:
        print(f"  {r['Group A']} vs {r['Group B']}: diff={r['Mean diff (A-B)']}, q={r['q-stat']}, p={r['p-adj']} {r['Sig']}")

posthoc_df = pd.DataFrame(posthoc_rows)[['Outcome','Group A','Group B',
                                          'Mean diff (A-B)','q-stat','p-adj','Sig']]
posthoc_df.to_csv(os.path.join(OUTPUT_DIR, 'results_posthoc.csv'), index=False)
print('\nSaved: results_posthoc.csv')


# ── 8. Figures ────────────────────────────────────────────────────────────────

sep('8. GENERATING FIGURES')

# Figure 1: Box plots for all four outcomes
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
fig.suptitle('Outcome Variables by Thinking Partner Condition', fontsize=13, fontweight='bold', y=1.01)

for ax, (outcome_col, outcome_label) in zip(axes, OUTCOMES.items()):
    data_by_cond = [df[df['condition_label'] == c][outcome_col].dropna().values
                    for c in CONDITION_ORDER]
    bp = ax.boxplot(data_by_cond, patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], CONDITION_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(CONDITION_ORDER, fontsize=12)
    ax.set_title(outcome_label, fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
path1 = os.path.join(FIGURES_DIR, 'fig1_boxplots.png')
plt.savefig(path1, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {path1}')

# Figure 2: Mean + SD bar chart
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle('Mean Scores by Condition (error bars = ±1 SD)', fontsize=13, fontweight='bold', y=1.01)

for ax, (outcome_col, outcome_label) in zip(axes, OUTCOMES.items()):
    means = [df[df['condition_label'] == c][outcome_col].mean() for c in CONDITION_ORDER]
    sds   = [df[df['condition_label'] == c][outcome_col].std(ddof=1) for c in CONDITION_ORDER]
    x = np.arange(len(CONDITION_ORDER))
    bars = ax.bar(x, means, yerr=sds, capsize=5, color=CONDITION_COLORS,
                  alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER, fontsize=12)
    ax.set_title(outcome_label, fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add mean value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
path2 = os.path.join(FIGURES_DIR, 'fig2_means_sd.png')
plt.savefig(path2, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {path2}')

# Figure 3: Distribution plots (violin + strip)
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
fig.suptitle('Score Distributions by Condition', fontsize=13, fontweight='bold', y=1.01)

for ax, (outcome_col, outcome_label) in zip(axes, OUTCOMES.items()):
    data_by_cond = [df[df['condition_label'] == c][outcome_col].dropna().values
                    for c in CONDITION_ORDER]
    parts = ax.violinplot(data_by_cond, positions=[1,2,3],
                          showmedians=True, showextrema=True)
    for i, (pc, color) in enumerate(zip(parts['bodies'], CONDITION_COLORS)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    # Overlay individual points with jitter
    for i, (vals, color) in enumerate(zip(data_by_cond, CONDITION_COLORS)):
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i+1) + jitter, vals,
                   alpha=0.4, s=8, color=color, zorder=3)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(CONDITION_ORDER, fontsize=12)
    ax.set_title(outcome_label, fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
path3 = os.path.join(FIGURES_DIR, 'fig3_violin.png')
plt.savefig(path3, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {path3}')

# Figure 4: Pre vs Post opinion by condition
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(CONDITION_ORDER))
width = 0.35
pre_means  = [df[df['condition_label'] == c]['pre_opinion_1_to_7'].mean() for c in CONDITION_ORDER]
post_means = [df[df['condition_label'] == c]['post_opinion_1_to_7'].mean() for c in CONDITION_ORDER]
pre_sds    = [df[df['condition_label'] == c]['pre_opinion_1_to_7'].std(ddof=1) for c in CONDITION_ORDER]
post_sds   = [df[df['condition_label'] == c]['post_opinion_1_to_7'].std(ddof=1) for c in CONDITION_ORDER]

bars1 = ax.bar(x - width/2, pre_means, width, yerr=pre_sds, capsize=4,
               label='Pre-discussion', color='#BBBBDD', edgecolor='white', alpha=0.9)
bars2 = ax.bar(x + width/2, post_means, width, yerr=post_sds, capsize=4,
               label='Post-discussion', color=CONDITION_COLORS, edgecolor='white', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(CONDITION_ORDER, fontsize=12)
ax.set_ylabel('Mean Opinion Score (1–7)', fontsize=13)
ax.set_title('Pre vs Post Opinion Score by Condition', fontsize=11, fontweight='bold')
ax.set_ylim(0, 8)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
path4 = os.path.join(FIGURES_DIR, 'fig4_pre_post_opinion.png')
plt.savefig(path4, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {path4}')

sep('DONE')
print('All outputs saved to current directory and figures/ subfolder.')
print('\nFiles generated:')
for f in ['analysis_clean.csv','results_descriptives.csv','results_assumptions.csv',
          'results_anova.csv','results_kruskal.csv','results_posthoc.csv']:
    print(f'  {f}')
print('  figures/fig1_boxplots.png')
print('  figures/fig2_means_sd.png')
print('  figures/fig3_violin.png')
print('  figures/fig4_pre_post_opinion.png')
