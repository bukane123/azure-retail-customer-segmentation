# Azure Retail Customer Segmentation

## Project Overview

This project builds a production-ready customer segmentation solution using
Azure Machine Learning, clustering algorithms, MLflow and batch inference.

The objective is to identify meaningful groups of retail customers based on
their purchasing behaviour so that the business can better understand and
engage different customer groups.

## Business Problem

Retail customers behave differently. Some customers purchase frequently,
some spend significantly more than others, some are new customers, and some
may have stopped purchasing.

Treating every customer in the same way can lead to poorly targeted marketing,
inefficient customer engagement and missed retention opportunities.

The business therefore needs a data-driven way to group customers with similar
behaviour.

## Business Objective

Build a customer segmentation model that groups customers according to their
transaction behaviour.

The resulting segments should be:

- distinct enough to describe meaningful differences in customer behaviour;
- understandable by business users;
- large enough to support useful business actions;
- stable enough to be used repeatedly as new transaction data becomes available;
- reproducible through an automated Azure Machine Learning pipeline.

## Machine Learning Problem

This is an unsupervised machine learning problem.

There is no existing target column that identifies the correct customer
segment.

Clustering algorithms will therefore be used to discover naturally occurring
groups of customers based on behavioural features.

Candidate algorithms may include:

- K-Means
- Gaussian Mixture Models
- Hierarchical Clustering
- DBSCAN

## Initial Customer Features

Customer-level features will be engineered from transaction data.

Initial candidate features include:

- Recency
- Frequency
- Monetary Value
- Average Order Value
- Number of Products Purchased
- Customer Tenure
- Purchase Frequency

The final feature set will be determined during exploratory analysis and
feature engineering.

## Model Evaluation

Because clustering has no known target label, traditional classification
metrics such as accuracy cannot be used.

Candidate clustering solutions will be evaluated using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- cluster stability
- cluster size and distribution
- business interpretability

The final model will not be selected solely from one numerical metric.

## Production Architecture

The project will eventually include:

Raw transaction data
→ data validation
→ customer feature engineering
→ preprocessing
→ clustering model training
→ model evaluation
→ MLflow experiment tracking
→ model registration
→ Azure ML pipeline
→ batch customer segmentation
→ monitoring and retraining

## Technology

- Python
- pandas
- scikit-learn
- Azure Machine Learning
- MLflow
- Azure ML Data Assets
- Azure ML Pipelines
- Azure ML Batch Endpoints
- Git and GitHub

## Project Status

Project setup and business problem definition.