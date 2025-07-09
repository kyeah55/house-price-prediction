# House Price Prediction

Project focuses on predicting house prices using machine learning techniques. The model is trained on housing data and produces predictions on unseen test data.

## Dataset

The dataset used is from Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

## Reqs

- pandas
- numpy
- scikit-learn
- joblib

Install dependencies with:

    pip install -r requirements.txt

## How to Run

Train the model:

    python src/model_training.py

Make predictions:

    python src/model_testing.py

(Optional) Evaluate the model:

    python src/model_evaluation.py

## Outputs

- models/best_model.pkl: Trained model  
- outputs/submission.csv: Predictions based on test data

## Notes

- This project was created for learning and practicing regression modeling.  
- `random_work.py` is a scratch file for experimentation.
