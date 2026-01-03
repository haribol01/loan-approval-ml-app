# Loan Approval ML Application

This project is an end-to-end machine learning application designed to predict loan approval decisions based on applicant financial and credit attributes.

The core model is a logistic regression implemented from scratch using gradient descent, without relying on machine learning libraries. Features such as CIBIL score, annual income, loan amount, number of dependents, and loan term are used to estimate approval probability while balancing precision and recall.

The trained model is exposed through a FastAPI backend, enabling real-time predictions via a REST API. A modern, responsive frontend built with HTML, Tailwind CSS, and JavaScript allows users to interact with the model seamlessly.

## Key Highlights

Logistic regression implemented from scratch using NumPy

Feature engineering

Precision–recall analysis

FastAPI backend

Responsive frontend with input validation and loading states

End-to-end ML workflow from data to deployment