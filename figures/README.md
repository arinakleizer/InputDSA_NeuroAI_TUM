# Figures

This folder holds all figures produced during the project.

| File | What it shows |
| --- | --- |
| `distance_heatmap.png` | Pairwise angular Procrustes distance between the 20 A matrices. Task blocks are visible as the strongest structure. |
| `dendrogram.png` | Hierarchical clustering of the same distance matrix. First split by task, second (inside PDM) by architecture. |
| `interaction_vs_rank_dense.png` | Interaction effect (DC arch effect minus PDM arch effect) as a function of the fit rank. Crosses zero near the AIC-optimal rank. |
| `B_distance_PerceptualDecisionMaking.png` | B B^T pairwise distance on PDM. Clear architecture block pattern. |
| `B_distance_DelayComparison.png` | B B^T pairwise distance on DC. No architecture block pattern. |
| `B_arch_effect_vs_rank.png` | Architecture effect on B as a function of rank. Stable large on PDM, near zero on DC across all ranks. |
| `B_singular_spectrum.png` | B singular value spectra on PDM and DC. GRU spectrum drops more steeply on PDM. |
| `B_input_modes.png` | Input mode loadings on PDM for vanilla and GRU. Shows the segregation vs blending mechanism. |
| `B_input_modes_lstm_pdm.png` | Input mode loadings on PDM for LSTM. Same segregation as GRU. |

## Additional figures

| File | What it shows |
| --- | --- |
| `distance_heatmap_rank_pr.png` | A distance heatmap at the participation ratio rank (rank 8). |
| `dendrogram_rank_pr.png` | Dendrogram at rank 8. |
| `FINAL_heatmap_rank15.png` | A distance heatmap at the AIC-optimal rank 15. |
| `FINAL_comparison_bar.png` | Bar chart comparison of task and architecture effects. |
| `distance_by_category.png` | Distances split into within-cell, across-arch, across-task categories. |
| `mds_projection.png` | 2D MDS projection of the 20 A matrices. |
| `mds_shepard.png` | Shepard diagram checking MDS distortion. |
| `permutation_test.png` | Permutation test null distributions for task and architecture effects. |
| `participation_ratio.png` | Participation ratio of hidden state covariance across models. |
| `optimal_hard_threshold.png` | Optimal Hard Threshold rank estimate (Gavish and Donoho 2014). |
| `mase_aic_sweep.png` | MASE and AIC as a function of rank, used to pick rank 15. |
| `interaction_hypothesis_test.png` | Hypothesis test for the A interaction at the original rank. |
| `sensitivity_hyperparams.png` | Sensitivity of the main effects to InputDSA hyperparameters. |
| `sensitivity_rank_pr.png` | Sensitivity check at the participation ratio rank. |
| `sensitivity_val_batch.png` | Sensitivity to the choice of validation batch. |
| `sensitivity_fixation.png` | Effect of removing the fixation channel from PDM. |
| `sensitivity_fixation_outlier_highlight.png` | Same, with the outlier seed highlighted. |
| `coarse_vs_fine_test.png` | Check on coarse vs fine dynamical structure. |
| `nestedness_check.png` | Check that the observed structure is not just nesting. |
| `outlier_eigenvalues.png` | Eigenvalues of the vanilla DC outlier seed. |
| `outlier_hidden_activity.png` | Hidden state activity of the outlier seed. |
| `vanilla_dc_fixed_points.png` | Fixed point analysis of vanilla RNN on DC. |
| `B_distance_no_fixation.png` | B B^T distance with fixation channel removed. Effect drops. |
| `B_effect_decomposition.png` | Decomposition of the B effect by input channel subset. |
| `B_effective_input_dim.png` | Effective input dimension (participation ratio of B singular values). |
| `B_loadings_both_tasks.png` | Loading heatmaps for PDM and DC side by side. |
| `B_channel_injection_angles.png` | Angles between raw input channel injection directions. |
| `B_state_angles.png` | Angles between input modes in state space. |
