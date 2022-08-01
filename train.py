import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from utils import generate_features


def train(train_csv_path: str, model_save_path: str, random_state: int):
    # loading data
    print(f'Loading dataset: {train_csv_path}...')
    train_df = pd.read_csv(train_csv_path)

    # retrieving embeddings
    embeddings = []
    for i, row in tqdm(train_df.iterrows(), 'Retrieving excerpt embeddings...'):
        excerpt_embed = generate_features(row['excerpt'])
        embeddings.append(excerpt_embed)

    embeddings = np.array(embeddings)
    target = train_df['target']

    # train-test split
    print('Splitting data into train and test sets...')
    X_train, X_test, y_train, y_test = train_test_split(embeddings, target, test_size=0.2,
                                                        random_state=random_state)
    print(f'--Train size: {len(X_train)}, Test size: {len(X_test)}')

    # model training
    print('Training RandomForestRegressor...')
    rf_model = RandomForestRegressor(random_state=random_state)
    rf_model.fit(X_train, y_train)

    # model evaluation
    print('Evaluating in test set...')
    y_preds = rf_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_preds, squared=False)
    print('--RMSE={:.4f}'.format(rmse))

    # save model
    joblib.dump(rf_model, model_save_path)
    print(f'Model saved to: {model_save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_path', type=str, required=True, help='Path to train data')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save model, .joblib format')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    train(train_csv_path=args.train_csv_path, model_save_path=args.model_save_path,
          random_state=args.random_state)