# Kaggle-Readability-Challenge

## Description
Simple model training pipeline, that consists of:
- transforming text excerpts into vectors using {all-MiniLM-L6-v2}(https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence transformer. Excerpt vector is an average from sentence vectors.
- splitting data into train and test parts
- training RandomForestRegressor with default parameters
- calculating RMSE on test set
- saving model

## Usage

1. Install requirements:

```pip install -r requirements.txt```

2. Download data from {CommonLit Readability Challenge}(https://www.kaggle.com/competitions/commonlitreadabilityprize/data) into folder `data/`


3. To train and evaluate model. Scripts runs up to 10 mins on CPU, final RMSE on test set is 0.7722. Model will be saved to `rf_model.joblib`.
```
python train.py 
--train_csv_path data/train.csv
--model_save_path rf_model.joblib
--random_state 42
```
