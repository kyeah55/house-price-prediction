# House Price Prediction

Project focuses on predicting house prices using machine learning techniques. The model is trained on housing data and produces predictions on unseen test data.

## Project Structure

project-folder/
│
├── data/
│   ├── train.csv             # Training dataset
│   └── test.csv              # Test dataset
│
├── models/
│   └── best_model.pkl        # Final trained model
│
├── outputs/
│   └── submission.csv        # Predictions ready for submission
│
├── src/
│   ├── model_training.py     # Training the model
│   ├── model_testing.py      # Generating predictions
│   ├── model_evaluation.py   # Model evaluation and comparison (optional)
│   └── random_work.py        # My work here(to find best values, columns to drop etc. You can check if you want but this file has no affect in project)
│
└── README.md                 # Project description (this file)

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