import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Define the feature ranges
feature_ranges = {
    'dti': (0, 400, 'Debt-to-Income Ratio', 'It is the ratio of a borrower\'s total monthly debt payments to their gross monthly income. It\'s a measure of a person\'s ability to manage monthly payments and repay debts. Lenders typically prefer a lower DTI, as it indicates that the borrower has more income available to cover their debt obligations.'),
    'revol_bal': (0, 1800000, 'Total Credit Revolving Balance', 'This represents the total outstanding balance on a borrower\'s revolving credit accounts (e.g., credit cards) at a specific point in time. Generally, a lower revolving balance is preferred, as it suggests that the borrower is not heavily reliant on credit and may have better control over their finances.'),
    'revol_util': (0, 100, 'Revolving Line Utilization Rate', 'It is the percentage of a borrower\'s total revolving credit limit that is currently being used. It reflects how much of the available credit is being utilized. Lenders usually prefer a lower revolving utilization rate, as a higher percentage may indicate a higher risk of default.'),
    'int_rate': (5, 35, 'Interest Rate', 'The interest rate is the cost of borrowing, expressed as a percentage of the loan amount. Borrowers generally prefer a lower interest rate, as it reduces the overall cost of the loan. Lenders may offer lower rates to borrowers with good credit and financial stability.'),
    'installment': (24, 1600, 'Monthly Installment', 'It is the fixed monthly payment a borrower makes to repay a loan. Lenders prefer a monthly installment that the borrower can comfortably afford, based on their income and other financial obligations.'),
    'annual_inc': (5000.0, 9000000, 'Annual Income', 'This is the total income earned by the borrower in a year. Lenders usually prefer a higher annual income, as it indicates the borrower\'s ability to cover loan payments and manage their financial responsibilities.'),
    'total_acc': (2, 160, 'Total Number of Credit Lines', 'It represents the total number of credit lines (credit cards, loans, etc.) the borrower has. While there\'s no specific preference for a higher or lower total number of credit lines, having a reasonable number and managing them responsibly is generally viewed positively by lenders.'),
    'loan_amnt': (1000, 40000, 'Loan Amount Requested', 'It is the amount of money the borrower is requesting as a loan. Lenders evaluate the loan amount in relation to the borrower\'s income and financial profile. They may prefer a reasonable loan amount that aligns with the borrower\'s ability to repay.'),
    'open_acc': (1, 90, 'Number of Open Credit Lines', 'It represents the number of currently active credit lines. Similar to the total number of credit lines, having a reasonable number and managing them responsibly is generally viewed positively by lenders.'),
    'mort_acc': (0, 40, 'Number of Mortgage Accounts', 'It indicates the number of mortgage accounts the borrower has. Having a mortgage account may be viewed positively, especially if the borrower has a history of making timely payments. It adds to the borrower\'s credit history and can be considered a positive factor.')
}

# Function to predict class label and probabilities
def predict_loan_status(features_input):
    # Load the saved model and scaler
    loaded_model = joblib.load('logreg_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')

    # Preprocess the input features
    features_input_scaled = loaded_scaler.transform(np.array(features_input).reshape(1, -1))

    # Make predictions
    predicted_class = loaded_model.predict(features_input_scaled)
    predicted_probabilities = loaded_model.predict_proba(features_input_scaled)

    # Display results
    st.write("\nPrediction Results:")
    
    if predicted_class[0] == 0:
        st.write(f"Applicant is predicted to be successful in paying the loan with a probability of {predicted_probabilities[0][0]*100:.2f}%.")
    else:
        st.write(f"Applicant is predicted to be at risk of defaulting on the loan with a probability of {predicted_probabilities[0][1]*100:.2f}%. Further assessment is recommended.")

    # Plot probabilities for class 0 (success) and class 1 (default)
    fig, ax = plt.subplots()
    classes = ['Successful Payment', 'Default']
    ax.bar(classes, predicted_probabilities[0], color=['green', 'red'])
    ax.set_ylabel('Probability')
    ax.set_title('Probabilities of Loan Status')
    st.pyplot(fig)

    # Explain the model using SHAP values
    explainer = shap.Explainer(loaded_model, loaded_scaler.transform(np.zeros((1, len(features_input)))))
    shap_values = explainer.shap_values(features_input_scaled)
    
    # Summary plot of feature importance using SHAP
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features_input_scaled, feature_names=list(feature_ranges.keys()), show=False)
    ax.set_title('Summary Plot of Feature Importance (SHAP Values)')
    st.pyplot(fig)

    # Display SHAP summary text
    st.write("\n**SHAP Summary:**")
    st.write("SHAP (SHapley Additive exPlanations) values provide insights into feature importance. Positive values contribute to the prediction of class 1 (default), while negative values contribute to the prediction of class 0 (successful payment).")

def app():
    st.title("Loan Status Prediction App")
    st.write("Enter the following details to predict the loan status:")

    # Display sidebar with feature explanations
    st.sidebar.title("Feature Explanations")
    for feature, (_, _, description, explanation) in feature_ranges.items():
        st.sidebar.subheader(feature)
        st.sidebar.write(description)
        st.sidebar.write(explanation)

    # Get user input
    user_input = {}
    for feature, (min_value, max_value, _, _) in feature_ranges.items():
        while True:
            try:
                value = st.number_input(f"{feature} ({min_value} to {max_value})", float(min_value), float(max_value))
                if min_value <= value <= max_value:
                    user_input[feature] = value
                    break
                else:
                    st.warning(f"Invalid input. Value should be between {min_value} and {max_value}.")
            except ValueError:
                st.warning("Invalid input. Please enter a numeric value.")

    # Predict loan status when "Predict Results" button is clicked
    if st.button("Predict Results"):
        predict_loan_status([user_input[feature] for feature in feature_ranges.keys()])

if __name__ == "__main__":
    app()
