# Bank Marketing Subscription Prediction

## Question

Can customer information predict whether a person will subscribe to a term deposit?

This project compares binary classification models that predict whether a bank customer subscribes to a term deposit (`yes`) or does not subscribe (`no`). It also examines false positives and false negatives to understand where the selected model makes mistakes.

## Dataset

The dataset comes from the [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).

- **File used:** `bank-additional-full.csv`
- **Target:** `y`, where `yes` means the customer subscribed and `no` means they did not.
- **Excluded feature:** `duration` was removed before training. It records the length of the call and would not be available when making a prediction before the call.

The dataset contains more customers who did not subscribe than customers who did. This class imbalance makes metrics such as recall and F1 important alongside precision.

## Setup and running the notebook

The notebook was developed with Python **3.12.6**.

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open **`src/train.ipynb`** and run all cells from top to bottom. The dataset should be placed at **`data/bank-additional/bank-additional-full.csv`** (relative to the notebook in `src/`).

## Methods

### Data preparation

The notebook inspects the dataset’s columns, data types, target distribution, missing values, and categorical features. It removes `duration`, separates the features from the target, and encodes `y` as a binary value (`yes` = 1, `no` = 0).

### Data splitting

The data is split into training, validation, and test sets using stratification so that each split has a similar proportion of subscribers.

- Training set: **60% (24,712 rows)**
- Validation set: **20% (8,238 rows)**
- Test set: **20% (8,238 rows)**
- Random seed: **42**

The training set is used to fit models. The validation set is used to compare models and select any tuning settings. The test set is reserved for one final evaluation after selecting the model and settings.

### Preprocessing and models

Categorical features are one-hot encoded. Numeric and categorical features are imputed if needed, and numeric features are scaled for logistic regression. Preprocessing is included in model pipelines so it is fitted on training data.

The notebook compares:

- **Majority-class baseline:** predicts the most common class.
- **Logistic regression**
- **Random forest**

Logistic regression is tuned using the validation set. Thresholds from 0.10 to 0.90 were compared on precision, recall, and F1. The selected threshold was **0.30**, chosen using validation data to improve recall while keeping precision usable.

### Evaluation metrics

The models are evaluated using:

- **Precision:** the share of predicted subscribers who actually subscribed.
- **Recall:** the share of actual subscribers the model identified.
- **F1:** the harmonic mean of precision and recall.
- **ROC-AUC:** how well the model ranks subscribers above non-subscribers across thresholds.
- **Confusion matrix:** counts of correct and incorrect predictions for each class.

Accuracy is not reported because non-subscribers are the majority class. A model can achieve high accuracy by predicting “no” for nearly everyone while failing to find subscribers.

## Results

### Validation-set comparison

The following values come from the validation-set evaluation at the default threshold of 0.50.

| Model               | Precision | Recall |    F1 |
| ------------------- | --------: | -----: | ----: |
| Majority baseline   |     0.000 |  0.000 | 0.000 |
| Logistic regression |     0.677 |  0.242 | 0.358 |
| Random forest       |     0.554 |  0.285 | 0.378 |

On the validation set, logistic regression had higher precision, while random forest had higher recall and F1. This suggests a trade-off: random forest identified a larger share of subscribers but produced more false positive predictions relative to its positive predictions.

### Final test-set results

| Selected model      | Threshold | Precision | Recall |    F1 | ROC-AUC |
| ------------------- | --------: | --------: | -----: | ----: | ------: |
| Logistic regression |      0.30 |     0.495 |  0.436 | 0.464 |   0.801 |

Test confusion matrix:

|                |      Predicted no |     Predicted yes |
| -------------- | ----------------: | ----------------: |
| **Actual no**  |              6897 |               413 |
| **Actual yes** |               523 |               405 |

The test set was used for this final evaluation only, after model and threshold selection were complete.

## Plots

The notebook includes at least two plots:

1. **Validation performance by model:** compares precision, recall, and F1 for the majority baseline, logistic regression, and random forest on the validation set.
2. **Test confusion matrix:** shows the final selected model’s true negatives, false positives, false negatives, and true positives on the test set.

The validation comparison chart is generated in **`src/train.ipynb`** as the bar plot titled “Validation performance by model”.

## Error analysis

The notebook inspects false positives and false negatives made by the selected model on the test set. It compares error cases across customer or campaign characteristics and reports group sizes alongside rates where possible.

Two patterns from the test-set analysis:

1. **False positives with prior success:** 60 of 413 false positives (14.5%) had `poutcome = success`, compared with 246 of 8,238 test customers (3.0%). The model often predicted subscription for customers whose previous campaign succeeded but who did not subscribe this time.
2. **False negatives in May:** 153 of 523 false negatives (29.3%) were contacted in May, compared with 175 of 928 actual subscribers (18.9%). The model missed a disproportionate share of May subscribers.

These are patterns in this dataset, not evidence that the features caused the errors. Avoid drawing conclusions from small groups without noting their size.

## Limitations

- **Class imbalance:** subscribers are less common than non-subscribers, so accuracy alone can hide poor performance on the subscriber class.
- **Dataset age:** the campaign data may not represent current customer behavior or current marketing practices.
- **Population differences:** results may change for another bank, region, time period, or customer population.
- **Error patterns are descriptive:** subgroup differences in model errors do not establish why those errors occurred.

## Reproducibility

To reproduce the analysis:

1. Use the dataset version and CSV filename listed above.
2. Install the packages listed in **Setup and running the notebook**.
3. Run all cells in **`src/train.ipynb`** from top to bottom.
4. Use random seed **42** and the documented split proportions (60% / 20% / 20%).
5. Keep `duration` excluded and make model and threshold selections using validation data only.

## Conclusion

On the validation set, **logistic regression** at threshold **0.30** achieved a recall of **0.455** and precision of **0.531**, compared with the majority baseline’s recall of **0.000**. On the held-out test set, its precision was **0.495**, recall was **0.436**, and F1 was **0.464**.

These results indicate that lowering the threshold from 0.50 to 0.30 helped the model find more subscribers, but at the cost of more false alarms. They should be interpreted with the dataset’s class imbalance, age, and possible differences from future customer populations in mind.
