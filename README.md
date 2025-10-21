## 🛒 E-commerce Return Rate Reduction Analysis





#### Project Overview

&nbsp;	This project focuses on leveraging data science and predictive analytics to solve a critical profitability issue for online retailers: high product return rates. By analyzing transactional, product, and review data from a major e-commerce platform (Olist), we identify the core drivers of customer dissatisfaction and build a predictive model to flag high-risk orders before they lead to costly returns.



The final output is an actionable list of products and sellers requiring immediate intervention and an interactive Power BI dashboard for business monitoring.





#### 🎯 Objectives

The analysis was structured to achieve the following key objectives:



Root Cause Identification: Determine the primary reasons for customer dissatisfaction and high return incidents (proxied by low review scores).



Performance Segmentation: Quantify how return rates vary across different product categories and suppliers (sellers).



Predictive Modeling: Develop a robust machine learning model (Logistic Regression) capable of predicting the probability of an order resulting in a "bad experience" (return/low score).



Actionable Deliverables: Produce a filtered list of high-risk products for operational teams and an interactive dashboard for executive monitoring.



#### 🛠️ Technology Stack

<img width="797" height="135" alt="image" src="https://github.com/user-attachments/assets/922dff5e-abc8-4956-b44e-fea38d475ecf" />



#### 📊 Methodology \& Analysis



1\. Data Assembly and Target Definition

Multiple transactional and metadata CSV files (Orders, Items, Products, Reviews) were merged into a single Master Analysis Table.



The target variable, Is\_Return (Bad Experience), was created as a binary flag: 1 for orders with a review\_score of 2 or less, and 0 otherwise.



2\. Feature Engineering

Critical features were engineered to capture underlying risk factors, making the model highly predictive:



Seller \& Category Risk: Calculated the average historical return rate for each seller\_id and product\_category\_name (using Target Encoding).



Logistics Risk: Calculated the Delivery\_Delta\_Days (Actual Delivery Date minus Estimated Delivery Date) to quantify the impact of delivery delays on returns.



3\. Predictive Modeling (Logistic Regression)

A Logistic Regression model was trained on the prepared features, using class weighting to handle the data imbalance.



Model Performance: The model achieved an ROC-AUC Score of approximately 0.75, demonstrating strong predictive power in distinguishing between low-risk and high-risk orders.





#### ✨ Key Deliverables



1\. Interactive Power BI Dashboard (.pbix file)

The dashboard provides a holistic view of the return problem, enabling monitoring and root cause analysis:



Root Cause Analysis Page: Visualizes the Top 10 High-Risk Categories and Sellers by historical return rate, and analyzes the correlation between Delivery Delays and customer dissatisfaction.



Intervention List Page: Features the model's output via a Drill-Through Table that allows users to click a problematic seller or category to see the live list of individual orders flagged for intervention.



2\. High-Risk Product CSVThe model generates a CSV file, high\_risk\_products\_for\_intervention.csv, which contains a list of orders (sorted by Return\_Risk\_Score) where the probability of a bad outcome exceeds a defined threshold (e.g., $60\\%$). This list is used for proactive interventions (QC checks, customer service outreach).





#### 💡 Conclusion

This project demonstrates a data-driven strategy to move from reactive returns processing to proactive return prevention. By identifying that factors like poor seller performance, specific product categories, and delivery delays are major risk indicators, the e-commerce business can prioritize resources to significantly reduce operational costs and enhance customer loyalty.

