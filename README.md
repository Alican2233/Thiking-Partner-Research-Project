# Can AI Make a "Thinking Partner" for Young Adults?

This repository contains the full experimental pipeline and analysis code for the CSE3000 Research Project:

> **"Can AI Make a 'Thinking Partner' for Young Adults? Fostering Responsible Opinion Formation Among Young Adults in the Age of Generative AI"**
> Alican Ekşi — EEMCS, Delft University of Technology, 2026

The study investigates how three AI thinking partner styles — **Steelman**, **Socratic**, and **Neutral** — affect opinion change, epistemic trust, and epistemic autonomy in simulated young adult personas. 159 simulated personas (53 per condition) each engaged in a five-exchange session with an LLM-based thinking partner on the topic of individual versus systemic responsibility for climate change, completing pre- and post-session surveys to measure the three outcome variables.

---

## Repository Structure

```
├── main/
|   └── experiment/
│       └── [experiment scripts]       # Pipeline to run the thinking partner sessions
|    └── analysis/
│       └── analysis.py                # Quantitative and qualitative analysis script
│
└── results/
    ├── figures/                   # All generated figures (boxplots, pre/post chart, etc.)
    ├── results/                   # ANOVA, post-hoc, Kruskal-Wallis, descriptives CSVs
    ├── survey_scores.csv          # Raw quantitative survey results per persona
    ├── qualitative_coded.csv      # Coded open-text responses with theme labels
    └── transcripts/               # Full conversation transcripts per persona session
```

---

## Requirements

Install the required Python packages:

```bash
pip install pandas scipy matplotlib
```

The experiment pipeline additionally requires [Ollama](https://ollama.com) running locally with the LLaMA 3.1:8b model pulled:

```bash
ollama pull llama3.1:8b
```

---

## Running the Experiment

> **Note:** The full experiment (159 personas × 5 exchanges) takes approximately 5 hours on a standard CPU. Results from the original experiment run are already included in `results/survey_scores.csv` and `results/transcripts/`.

Navigate to the experiment directory and run:

```bash
cd experiment
python [experiment script name].py
```

Each persona session runs sequentially. If any session fails, an error file is written for that persona and the pipeline continues automatically. To rerun only failed or missing sessions, use:

```bash
python run_missing_personas.py
```

Session outputs are saved to `results/survey_scores.csv` (quantitative) and `results/transcripts/` (full conversation logs) as they complete.

---

## Running the Analysis

Place `survey_scores.csv` in the same directory as `analysis.py`, then run:

```bash
cd analysis
python analysis.py
```

The script will generate the following outputs in the `results/` directory:

**CSV files:**
- `results_descriptives.csv` — mean, SD, and median per condition per outcome variable
- `results_assumptions.csv` — Shapiro-Wilk and Levene's test results
- `results_anova.csv` — one-way ANOVA results for all three outcome variables
- `results_kruskal.csv` — Kruskal-Wallis robustness check results
- `results_posthoc.csv` — Tukey HSD post-hoc comparisons for opinion change
- `analysis_clean.csv` — full cleaned dataset with all computed scores

**Figures (saved to `results/figures/`):**
- `fig1_boxplots.png` — outcome variable distributions by condition
- `fig2_means_sd.png` — mean scores with standard deviation bars
- `fig3_violin.png` — violin plots of score distributions
- `fig4_pre_post_opinion.png` — pre vs post opinion scores by condition

---

## Models Used

| Component | Model |
|---|---|
| Thinking partners | LLaMA 3.1:8b (via Ollama API, localhost:11434) |
| Simulated personas | LLaMA 3.1 with 8.03B parameters |

---

## Results Overview

| Outcome | F(2, 152) | p-value | η² | Result |
|---|---|---|---|---|
| Opinion Change (SQ1) | 35.46 | < .001 | .318 | **Significant** |
| Epistemic Trust (SQ2) | 2.02 | .136 | .026 | Not significant |
| Epistemic Autonomy (SQ3) | 2.78 | .065 | .035 | Not significant (borderline) |

The Steelman condition produced a negative mean opinion shift (M = −0.42), moving personas away from individual climate action, while Socratic (M = +1.06) and Neutral (M = +1.37) produced comparable positive shifts. Post-hoc analysis confirmed Steelman differed significantly from both other conditions (p < .001), while Socratic and Neutral did not differ from each other (p = .373).

---

## Citation

If you use this code or data, please cite:

> Ekşi, A. (2026). *Can AI Make a "Thinking Partner" for Young Adults? Fostering Responsible Opinion Formation Among Young Adults in the Age of Generative AI.* BSc thesis, EEMCS, Delft University of Technology.

---

## Supervisors

Ujwal Gadiraju, Esra de Groot, Marije van Dalen, Shreyan Biswas — EEMCS, Delft University of Technology
