import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("loan_approval_dataset.csv")

f1,f2,f3,f4,f5 = data['cibil_score'], data['income_annum'], data['loan_amount'], data['no_of_dependents'], data['loan_term']
y = data['loan_status']

n = len(y)
x1,x2,x3,x4,x5 = (f1 - f1.min()) / f1.std(), (f2 - f2.min()) / f2.std(), (f3 - f3.min()) / f3.std(), (f4 - f4.min()) / f4.std(), (f5 - f5.min()) / f5.std()
w1 = w2 = w3 = w4 = w5 = b = 0
L = 0.1
epochs = 1000

for epoch in range(epochs):
    z = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + b
    y_pred = 1 / (1 + np.exp(-z))
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    dw1 = (1/n) * np.sum(x1 * (y_pred - y))
    dw2 = (1/n) * np.sum(x2 * (y_pred - y))
    dw3 = (1/n) * np.sum(x3 * (y_pred - y))
    dw4 = (1/n) * np.sum(x4 * (y_pred - y))
    dw5 = (1/n) * np.sum(x5 * (y_pred - y))
    db  = (1/n) * np.sum(y_pred - y)
    w1 -= L * dw1
    w2 -= L * dw2
    w3 -= L * dw3
    w4 -= L * dw4
    w5 -= L * dw5
    b -= L * db
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print(f"Trained weights: w1={w1}, w2={w2}, w3={w3}, w4={w4}, w5={w5}, b={b}")
print("Model training complete.")

def predict(cibil_score, income_annum, loan_amount, no_of_dependents, loan_term):
    x1_new = (cibil_score - f1.min()) / f1.std()
    x2_new = (income_annum - f2.min()) / f2.std()
    x3_new = (loan_amount - f3.min()) / f3.std()
    x4_new = (no_of_dependents - f4.min()) / f4.std()
    x5_new = (loan_term - f5.min()) / f5.std()
    z_new = w1 * x1_new + w2 * x2_new + w3 * x3_new + w4 * x4_new + w5 * x5_new + b
    y_new_pred = 1 / (1 + np.exp(-z_new))
    return 1 if y_new_pred >= 0.5 else 0

def test_model():
    test_data = pd.read_csv('test_data.csv')
    tp = tn = fp = fn = 0
    for index, row in test_data.iterrows():
        prediction = predict(row['cibil_score'], row['income_annum'], row['loan_amount'], row['no_of_dependents'], row['loan_term'])
        actual = row['loan_status']
        if prediction == 1 and actual == 1:
            tp += 1
        elif prediction == 0 and actual == 0:
            tn += 1
        elif prediction == 1 and actual == 0:
            fp += 1
        elif prediction == 0 and actual == 1:
            fn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

def client_interaction():
    isRunning = True
    while isRunning:
        print("1. Predict Loan Approval")
        print("2. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            cibil_score = float(input("Enter CIBIL Score: "))
            income_annum = float(input("Enter Annual Income: "))
            loan_amount = float(input("Enter Loan Amount: "))
            no_of_dependents = int(input("Enter Number of Dependents: "))
            loan_term = int(input("Enter Loan Term (in months): "))
            result = predict(cibil_score, income_annum, loan_amount, no_of_dependents, loan_term)
            if result == 1:
                print("Loan Approved")
            else:
                print("Loan Not Approved")
        elif choice == '2':
            isRunning = False

if __name__ == "__main__":
    client_interaction()    