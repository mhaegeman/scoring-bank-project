# Bank Scoring Dashboard Project
Welcome to the Bank Scoring Dashboard project repository! This project offers an innovative credit scoring tool designed to calculate the probability of a client repaying their loan, as well as an interactive dashboard aimed at enhancing customer relationships.

## Overview
The main goal of this project is to implement a credit scoring tool to evaluate the likelihood of a client repaying their loan. Additionally, we aim to develop an interactive dashboard dedicated to customer relationship management, providing bank advisors with valuable insights into clients' financial situations.

## Key Features
- *Credit Scoring Tool*: This tool calculates the probability of a client repaying their loan based on their financial data.
- *Interactive Dashboard*: An intuitive dashboard that allows bank advisors to visualize clients' financial data, thereby enhancing customer relationships.
- *Model Interpretability*: The project ensures that the model's predictions are understandable and easy to communicate, assisting bank advisors in their decision-making process.


## Challenges
The project addresses the following challenges:

- Cleaning and extracting important information from multiple datasets
- Training and optimizing a classification model with imbalanced datasets
- Extracting useful data for the development of an interactive customer relationship dashboard
- Deploying the dashboard online via a VPS

## Approach
The project follows a three-step approach:

1. *Data Import and EDA (Exploratory Data Analysis)* for each dataset:
- Handling missing values
- Analyzing correlations
- Performing feature engineering

2. *Training and Evaluation of the LGBMClassifier model*:
- Simple training
- OverSampling
- UnderSampling
- Custom metrics with threshold

3. *Model Interpretability*:
- Feature importance
- Selection of important variables for the dashboard

## Project Structure
The project is organized into two main parts: the scoring model and the interactive dashboard. The project files are as follows:

1. *Scoring Model Code*
- `functions.py`: Main functions used in the notebook
- `notebook_scoring.ipynb`: A notebook for exploration and modeling, including preprocessing steps adapted from an existing Kaggle kernel, which can be found [here](https://www.kaggle.com/code/ekrembayar/homecredit-default-risk-step-by-step-1st-notebook/notebook)

2. *Interactive Dashboard*
- `dashboard_app_streamlit.py`: The code for the dashboard, developed with Streamlit
- `notebook_prep_API.ipynb`: A file containing tests for the application and the creation of additional tables

## Getting Started
To begin using this project, follow these steps:

1. Clone the repository: `git clone https://github.com/yourusername/bank-scoring-dashboard.git`
2. Navigate to the project directory: `cd bank-scoring-dashboard`
3. Install the required dependencies: `pip install -r requirements.txt`
4 Run the Streamlit dashboard: `streamlit run dashboard_app_streamlit.py`

By following the above steps, you will have access to the interactive dashboard, which serves as a valuable tool for bank advisors. The dashboard is packed with various graphs and metrics that allow users to effectively evaluate a client's financial situation.

Feel free to explore the code, contribute to the project, and share your insights!

---
This project is a comprehensive and user-friendly solution for bank advisors looking to enhance their customer relationships and make informed decisions regarding loan approvals. With an easy-to-understand credit scoring tool and an interactive dashboard, bank advisors can efficiently evaluate clients' financial situations and provide valuable insights to their clients.
