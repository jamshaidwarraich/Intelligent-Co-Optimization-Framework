# Intelligent Co-Optimization Framework — BOHB Hyperparameter Search

This repository contains the **BOHB (Bayesian Optimization with Hyperband) hyperparameter search code** used in:

> **An Intelligent Co-Optimization Framework for Hyperparameters and Optimizer-Loss Configurations in Physics Informed Protein Folding Models**
> Jamshaid Ul Rahman, Iqra Noureen, Rongin Uwitije, Areen Rasool

---

## Relationship to the base PhyFold codebase

The underlying physics-informed neural network (PINN) architecture, the reaction-diffusion PDE formulation, and the optimizer-loss comparison code used in this study are **not redefined here**. They are already publicly available in our earlier PhyFold repository:

**🔗 https://github.com/jamshaidwarraich/PhyFold**

That repository implements the model described in:

> Ul Rahman, J., Noureen, I., Mannan, A., & Uwitije, R. (2026). *PhyFold: Environment-Aware Mathematical Modeling for Protein Folding Dynamics Integrated with Physics-Informed Neural Network.* Journal of Cheminformatics, 18, Article 71. https://doi.org/10.1186/s13321-026-01271-w

including the governing PDE, the composite physics-informed loss function, and the finite-difference validation of the PINN solution.

**This repository adds only the BOHB-specific component** built on top of that base code:

- Hyperparameter search worker
- Search space definition (Table 1 of the paper)
- Search execution script
- Resulting search logs

This corresponds to the *"BOHB Optimization"* and *"BOHB-PINN"* subsections of the paper.

The twelve optimizer-loss configurations compared in the paper (Tables 2 and 3, Section 5.3) are trained using the PhyFold base code above, with the architecture and learning rate fixed to the best configuration found by the BOHB search in this repository.

> **Coming soon:** The optimizer-loss comparison code (training the BOHB-selected configuration under twelve optimizer/loss-function combinations across five seeds, and the protein-disjoint cross-validation in Section 5.4) is being finalized and will be added shortly. Until then, it can be reconstructed from the base PhyFold code above by substituting the optimizer and data-loss function as described in the paper.

---

## What is in this repository

```
├── bohb/
│   ├── run_bohb_optimization.py   # Launches the BOHB search (NameServer, workers, BOHB master)
│   ├── pinn_worker.py             # BOHB worker: trains/evaluates one PINN configuration per trial
│   ├── pinn_model.py              # PINN architecture and PDE residual (shared with base PhyFold code)
│   ├── data_utils.py              # Embedding/label loading, train/val/test split, PDE domain setup
│   └── plot_bohb_results.py       # Diagnostic plots: incumbent trajectory, loss-vs-budget, loss-vs-hyperparameter
├── bohb_logs/
│   └── bohb_saved_phyfold_bohb_.../   # Full search history: all trials, incumbent trajectory, best configuration
├── notes/
│   └── bohb_config.md             # Exact search settings used for the paper
├── requirements.txt
└── README.md
```

---

## Search configuration used in the paper

| Setting | Value |
|---|---|
| Library | [HpBandSter](https://github.com/automl/HpBandSter) |
| Method | BOHB (Falkner et al., 2018, *ICML*) |
| Successive-halving parameter (η) | 3 |
| Budget range | 0.1 – 1.0 (→ 5,000 – 50,000 training epochs) |
| Parallel workers | 2 |
| Iterations | 60 |
| PDE collocation points (N_f) | 1500 (fixed during search) |
| Boundary points (N_bc) | 200 (fixed during search) |
| Loss weighting during search | All coefficients = 1 (uniform) |
| Search space | Hidden dimension ∈ {32, 64, 128, 256}; hidden layers ∈ {2, 3, 4}; learning rate ∈ [10⁻⁵, 10⁻³] (log-uniform); optimizer ∈ {Adam, RMSprop, AdamW} |
| Seeding | Base seed 1234; per-configuration seed = base seed + int(MD5(sorted config)[:8], 16), for reproducible-per-configuration, distinct-across-configuration initialization |

Full details are in [`notes/bohb_config.md`](notes/bohb_config.md).

---

## Note on data splits

The search in this repository uses the **standard 80/10/10 split** (`data_utils.py`, fixed seed 1234). The protein-disjoint evaluation reported in the paper (Section 5.4) is a separate, later step performed *after* the BOHB-selected configuration was fixed, and is not part of the search shown here.

---

## Reproducing the search

```bash
pip install -r requirements.txt

python bohb/run_bohb_optimization.py \
    --emb_path  data/kpro_embeddings.npy \
    --lab_path  data/kpro_labels.csv \
    --work_dir  bohb_results
```

Datasets (KPro, Rocklin) are not redistributed here; download links are given in the paper's Data Availability statement.

To generate diagnostic plots from a completed search:

```bash
python bohb/plot_bohb_results.py --saved_dir bohb_results/bohb_saved_phyfold_bohb_<timestamp>
```

---

## Citation

If you use this code, please cite **both** the present paper and the base PhyFold study.

**This paper:**
```bibtex
@article{rahman2026cooptimization,
  title={An Intelligent Co-Optimization Framework for Hyperparameters and Optimizer-Loss Configurations in Physics Informed Protein Folding Models},
  author={Rahman, Jamshaid Ul and Noureen, Iqra and Uwitije, Rongin and Rasool, Areen},
  journal={International Journal of Computational Intelligence Systems},
  year={2026}
}
```

**Base PhyFold study:**
```bibtex
@article{rahman2026phyfold,
  title={PhyFold: Environment-Aware Mathematical Modeling for Protein Folding Dynamics Integrated with Physics-Informed Neural Network},
  author={Rahman, Jamshaid Ul and Noureen, Iqra and Mannan, Abdul and Uwitije, Rongin},
  journal={Journal of Cheminformatics},
  year={2026},
  volume={18},
  pages={71},
  doi={10.1186/s13321-026-01271-w}
}
```

---

## License

[Add license here]





