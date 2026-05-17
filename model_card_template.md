# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random Forest Classifier trained using scikit-learn (v1.5.1). The model uses 100 estimators with a fixed random state of 42. Categorical features are one-hot encoded using a fitted `OneHotEncoder`; the binary income label is encoded using a `LabelBinarizer`. Both the model and encoder are serialized to disk via pickle.

## Intended Use

Predict whether an individual earns more than \$50,000 per year based on demographic and employment attributes from U.S. Census data. Intended for educational use as part of an ML pipeline deployment exercise.

## Training Data

The UCI Census Income dataset (`census.csv`) containing approximately 32,561 records. An 80/20 train-test split was applied (`random_state=42`), yielding roughly 26,048 training samples. Categorical features are one-hot encoded; continuous features are passed through as-is.

Categorical features used:
- workclass, education, marital-status, occupation, relationship, race, sex, native-country

## Evaluation Data

The held-out 20% test split (~6,513 samples) from the same census dataset, processed using the encoder fitted on training data only to prevent data leakage.

## Metrics

Performance on the test set:

| Metric    | Value  |
|-----------|--------|
| Precision | 0.7419 |
| Recall    | 0.6384 |
| F1 Score  | 0.6863 |

Per-slice performance across all categorical features is available in `slice_output.txt`.

## Ethical Considerations

The dataset includes sensitive demographic attributes such as race, sex, and native country, which are used as model features. Model performance varies across demographic slices. This model should not be used to make decisions that affect individuals, as it may reflect and reinforce historical biases present in the 1994 Census data.

## Caveats and Recommendations

- The data originates from the 1994 U.S. Census Bureau; income thresholds and demographic distributions have changed significantly since then.
- The model is not suitable for production use in any context involving consequential decisions about individuals.
- Review `slice_output.txt` before deployment to identify demographic groups where model performance is notably lower.
