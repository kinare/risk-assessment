import pandas as pd
import joblib

# Load the model and scalar
model = joblib.load('trained_model.joblib')
scalar = joblib.load('scalar.joblib')

# function to predict risk
def calculate_risk(industry, income, total_tax_filed):
    # Prepare the data for prediction
    industry_risk = {
        'Agriculture': 0,
        'Education': 1,
        'Healthcare': 2,
        'Services': 3,
        'Retail': 4,
        'Manufacturing': 5,
        'Construction': 6,
        'Finance': 7,
        'Technology': 8,
        'Energy': 9
    }

    industry_encoded = industry_risk.get(industry, 2)

    # Create a data frame for single prediction
    data = pd.DataFrame({
        'Industry': [industry_encoded],
        'Income': [income],
        'Total_Tax_Filed': [total_tax_filed]
    })

    # Scale the data using the same scaler used during training
    scaled_data = scalar.transform(data)

    # Predict the risk
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    return prediction[0], probability

if __name__ == '__main__':
    industry = input("Enter the industry: ").strip().capitalize()

    try:
        income = float(input("Enter the income: ").strip())
        total_taxes_filed = int(input("Enter the the number of taxes filed for the year: ").strip())
    except ValueError:
        print("Invalid input. Please enter a valid number for income and tax filings.")
        exit()

    prediction, probability = calculate_risk(industry, income, total_taxes_filed)

    print(f"The predicted risk for the industry '{industry}' is: {prediction}")
    print(f"High Risk: {'Yes' if prediction == 1 else 'No'}")
    print(f"Probability of being High Risk: {probability:.2f}")
