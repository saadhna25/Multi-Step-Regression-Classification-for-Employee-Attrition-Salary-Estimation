

Multi-Step Regression + Classification for Employee Attrition & Salary Estimation
(Attrition Risk & Salary Loss Prediction using Classification + Regression)

Project Description:
This project addresses employee attrition prediction and estimates the financial loss associated with employee exits. It is structured in multiple parts using machine learning techniques:

Part 1: Classification to predict if an employee will leave (Attrition model)

Part 2: Simulation of future salary based on performance increments

Part 3: Regression to predict future salary for employees likely to stay

Part 4: Selection of likely stayers using a probability threshold

Part 5: Calculation of total expected salary loss based on attrition probabilities and predicted salaries



Output Metrics:
Classification Metrics: Accuracy, F1-score, ROC-AUC

Regression Metrics: R² Score, RMSE

Financial Metric: Total Expected Salary Loss (₹)



Code Overview:
Part 1: Classification
Train-test split

Pipeline with preprocessing and SVM

Evaluation using F1-score and ROC-AUC

Part 2: Salary Simulation
Assign increment factor based on PerformanceRating

Simulate FutureSalary using the formula:

#FutureSalary = MonthlyIncome * Increment

Part 3: Regression on Likely Stayers
Predict FutureSalary using Random Forest Regressor

Trained only on employees with high P_stay (probability of staying > 0.6)

Part 4: Filtering
Use predicted probabilities from classification model

Select only likely stayers for regression training

Part 5: Financial Loss Estimation
Calculate:

#Expected Loss_i = P_leave_i * FutureSalary_i



                   ----------X----------X----------X---------


