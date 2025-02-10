import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# function to generate and prepare the dataset
def generate_and_prepare_data():
    industries = ['Agriculture', 'Manufacturing', 'Retail', 'Services', 'Technology', 'Healthcare', 'Construction', 'Finance', 'Education', 'Energy']
    income_ranges = {
        'Agriculture': (300000, 1500000),
        'Manufacturing': (500000, 2000000),
        'Retail': (300000, 1000000),
        'Services': (300000, 1500000),
        'Technology': (800000, 3000000),
        'Healthcare': (600000, 2500000),
        'Construction': (400000, 1800000),
        'Finance': (1000000, 4000000),
        'Education': (300000, 1200000),
        'Energy': (800000, 3500000)
    }
    # generate random data
    n_records = 200

    data = {
        'Industry': np.random.choice(industries, n_records),
        'Income': np.zeros(n_records, dtype=int),
        'Total_Tax_Filed': np.random.randint(1, 15, n_records)
    }

    for i in range(n_records):
        industry = data['Industry'][i]
        min_income, max_income = income_ranges[industry]
        data['Income'][i] = np.random.randint(min_income, max_income + 1)

    df = pd.DataFrame(data)

    # create a binary target variable for demonstration
    df['Risk'] = ((df['Income'] > 100000) & (df['Total_Tax_Filed'] < 6)).astype(int)

    #Convert Categorical data to numerical data
    le = LabelEncoder()
    df['Industry'] = le.fit_transform(df['Industry'])

    # Features and target variable
    x = df[['Industry', 'Income', 'Total_Tax_Filed']]
    y = df['Risk']

    return x, y

# function to train the model
def train_model(x, y):
    # Split the dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scalar = StandardScaler()
    x_train_scaled = scalar.fit_transform(x_train)
    x_test_scaled = scalar.transform(x_test)

    # Initialize the model
    model = LogisticRegression(random_state=42)
    model.fit(x_train_scaled, y_train)

    # Save the model and scalar
    joblib.dump(model, 'trained_model.joblib')
    joblib.dump(scalar, 'scalar.joblib')
    print("Model and scaler saved successfully!")

if __name__ == '__main__':
    x, y = generate_and_prepare_data()
    train_model(x, y)
