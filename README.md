# Breast Cancer Classification with K-Nearest Neighbors

**Machine Learning Theory · Assignment 1 · BCS 3B**  
**Author:** Gayatri Das · Individual submission  
**Roll ID:** `241001001195`

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devsbygayatri/knn-wisconsin-breast-cancer/blob/main/KNN_Wisconsin.ipynb)

[View the notebook on GitHub](KNN_Wisconsin.ipynb)

## Project objective

This project classifies breast cell samples as **benign** or **malignant** using K-Nearest Neighbors (KNN). The main objective is to catch as many malignant cases as possible while controlling false alarms. **Malignant recall** measures the share of actual malignant cases the model identifies, so it is the primary metric for model selection. Precision, accuracy, F1 and log loss provide additional views of performance.

## Dataset

The notebook downloads the original Wisconsin Breast Cancer dataset directly from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original).

| Dataset detail | Value |
|---|---|
| Records | 699 |
| Class distribution | 458 benign, 241 malignant |
| Input features | Nine cell measurements |
| Missing values | 16, all in Bare nuclei |
| Target encoding | Benign = 0, malignant = 1 |

The features include clump thickness, cell size and shape uniformity, and bare nuclei. The notebook uses sample IDs to keep repeated IDs together during cross-validation. IDs are excluded from the feature matrix.

## How to run the notebook

1. Click **Open in Colab** above.
2. Use a standard CPU runtime and run the first cell, which installs **scikit-learn 1.9.1**. If Colab asks you to restart the session after installation, restart before running the remaining cells.
3. Run the cells in order. The notebook downloads the dataset automatically.
4. Review the exploratory plots, held-out results, confusion matrix and log-loss chart. The last sections show the settings selected for the final fit and the effect of changing K.

The notebook fixes the cross-validation seed at `42` and pins scikit-learn because the version of its group splitter can affect which records appear in each fold.

## How the model works

1. **Explore the data.** The notebook displays descriptive statistics, class counts, missing-value counts and feature histograms.
2. **Prepare the features.** It reads `?` as a missing value and uses a pipeline with median imputation, standard scaling and KNN. Each fold learns its medians and scaling statistics from its own training rows.
3. **Classify by neighbour votes.** KNN finds the K closest training samples. Uniform voting gives each neighbour one vote, while distance voting gives closer neighbours more influence. The model predicts malignant when its malignant vote score reaches the selected cutoff.
4. **Tune and evaluate separately.** The five outer folds estimate performance. Within each outer training split, four inner folds compare settings. Rows with the same sample ID stay together at both levels. The notebook pools the five sets of held-out predictions so each record contributes once to the overall metrics.
5. **Fit the final model.** After evaluation, the notebook reruns the search on all 699 records and refits the selected pipeline. The held-out metrics describe the tuning procedure across the outer folds. The final fit uses all available data.

## Optimization

| Setting | Candidates |
|---|---|
| Neighbours, K | 5, 9, 15 |
| Distance | Manhattan and Euclidean |
| Voting | Uniform and distance weighted |
| Malignant cutoff | 0.25, 0.35, 0.45, 0.50 |

The grid search compares **48 configurations**. Among configurations that reach **95% mean inner-validation precision**, it selects the one with the highest mean malignant recall and uses precision to break ties. If none reaches the precision target, it ranks all candidates by recall and then precision. The precision target is a selection rule for the inner folds.

The saved final fit selected **K = 15**, **Manhattan distance**, **uniform voting** and a **0.25 cutoff**. The settings selected within individual outer folds can differ. The pipeline and parameter grid keep preprocessing and tuning in one workflow. This optimization concerns model settings and the classification decision rule. The project did not benchmark runtime speed.

## Results

These values are from the saved notebook's pooled held-out predictions.

| Metric | Result |
|---|---:|
| Accuracy | 96.85% |
| Malignant precision | 93.98% |
| **Malignant recall** | **97.10%** |
| Malignant F1 | 95.51% |
| Log loss | 0.3227 |

| Actual class | Predicted benign | Predicted malignant |
|---|---:|---:|
| Benign | 443 | 15 |
| Malignant | 7 | 234 |

The model caught **234 of 241** malignant cases and missed seven. It also flagged 15 benign cases as malignant. In a separate check using the same outer folds and preprocessing, a basic KNN with K = 5, uniform voting, Euclidean distance and a 0.50 cutoff caught 231 malignant cases and raised 13 false alarms. Tuning therefore caught **three additional malignant cases** at the cost of **two additional false alarms**. The notebook displays the tuned results. The basic KNN comparison comes from the separate check.

### Log loss

The notebook computes log loss from held-out probabilities before applying the cutoff. Lower values indicate that the model assigns more probability to the actual classes. Log loss ranged from **0.0728 to 0.5977** across the five folds, with a pooled value of **0.3227**. This variation adds context to the classification scores because log loss penalizes confident errors. Changing the cutoff can affect recall and precision while leaving the underlying probabilities unchanged.

### Overfitting and underfitting

The accuracy graph varies K while holding Euclidean distance, uniform voting and a 0.50 cutoff fixed. In a verification run with scikit-learn 1.9.1, the gap between mean training and validation accuracy narrowed from about **1.29 percentage points at K = 5** to **0.10 at K = 15**. There was no clear sign of underfitting across the three values examined. This graph checks the effect of K under fixed settings. The full search uses malignant recall to select the model.

## Project scope

The notebook covers data exploration, imputation, scaling, model tuning, grouped cross-validation and evaluation. KNN uses neighbour votes, so this implementation has no gradient optimizer, activation function or per-epoch training loss. Log loss is reported as an evaluation metric. The project uses the original records without synthetic augmentation and evaluates performance within the Wisconsin dataset. Probability calibration and external validation are outside the scope of this assignment.

## References

- [UCI Machine Learning Repository · Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

