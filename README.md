# Scania APS Fault Prediction: An End-to-End MLOps Project

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

**A complete MLOps project to predict Air Pressure System (APS) failures in Scania trucks, moving from reactive manual checks to a proactive, data-driven maintenance strategy. This solution is designed to minimize operational costs by focusing on a custom, business-centric evaluation metric.**

---

## 1. Executive Summary

Scania, a leader in the automotive industry, currently employs a manual, time-consuming, and cost-inefficient process for detecting critical faults in truck Air Pressure Systems (APS). This project introduces a predictive machine learning model that leverages existing sensor data from the vehicle's "black box" to proactively identify APS faults. The primary business driver is **cost reduction**, specifically by minimizing the high cost of missed detections (False Negatives), which are **50 times more expensive** than unnecessary checks (False Positives). The success of this initiative is measured by a single, clear metric: the total cost incurred from prediction errors.

## 2. The Business Opportunity

### The Core Problem
The Air Pressure System (APS) is a pivotal component of a truck's braking system, making its reliability a top safety priority. The current methodology for diagnosing APS faults is a major operational bottleneck:
- **Time-Consuming:** Technicians must manually inspect APS components on every truck that enters a service center.
- **Cost-Inefficient:** Significant labor and resources are expended inspecting healthy systems.
- **Reactive, Not Proactive:** Faults are only discovered after they have potentially developed, not before they become critical.

### The Proposed Solution
This project transforms the manual, reactive process into an automated, predictive one. We have developed a predictive model that uses sensor readings to accurately classify whether a truck has an APS fault. This ML-powered diagnostic tool integrates directly into the service center workflow, enabling a smarter, data-driven approach to maintenance.

The model is built on a robust dataset from over **36,000 Scania trucks**, pre-classified by domain experts into `pos` (fault) and `neg` (no fault) categories.

## 3. Financial Impact & Success Metric

The model's performance is directly tied to tangible business outcomes. The financial implications of prediction errors are clear and quantifiable.

### The Cost of Prediction Errors
*   **Cost 1: False Positive ($10)**
    *   **Scenario:** The model predicts a fault, but none exists.
    *   **Impact:** An unnecessary manual inspection is performed, incurring a nominal cost of **$10**.

*   **Cost 2: False Negative ($500)**
    *   **Scenario:** The model fails to detect a genuine APS fault.
    *   **Impact:** A truck with a faulty braking system leaves the service center, risking critical failures, extensive repairs, and brand damage. The estimated cost is **$500**.

### The Total Cost Formula
The primary Key Performance Indicator (KPI) is the **Total Cost of the Predictive Model**. Our objective is to minimize this value.
Total Cost = (Number of False Positives × $10) + (Number of False Negatives × $500)

---

## 4. Notebook Experimentations

A detailed experimentation phase was conducted in a Jupyter Notebook to understand the data intricacies and identify the most effective strategies.

### Exploratory Data Analysis (EDA) Insights
The initial analysis revealed several challenges:
1.  **Severe Class Imbalance:** The `neg` class (no fault) vastly outnumbered the `pos` class (fault).
2.  **Widespread Missing Values:** Many sensor columns contained a high percentage of `NaN` values.
3.  **Significant Outliers:** The data was characterized by right-skewed distributions and numerous outliers.

### The Winning Model Pipeline
After extensive experimentation using a `GridSearchCV` pipeline optimized for our custom cost metric, the best-performing model was identified:
-   **Scaler**: `RobustScaler()` to handle outliers effectively.
-   **Imputer**: `SimpleImputer(strategy='median')` for robust handling of missing data.
-   **Imbalance Handler**: `RandomUnderSampler()` to balance the class distribution.
-   **Classifier**: `XGBClassifier()` for its high performance and robustness.

---

## 5. MLOps Architecture Overview

This project is built around a robust MLOps architecture designed for automation, scalability, and reproducibility. The entire lifecycle of the machine learning model, from data ingestion to deployment, is orchestrated through a series of automated pipelines and cloud services.


**The core components of the architecture are:**
- **Modular Codebase:** The project is structured as an installable Python package, promoting code reusability and maintainability.
- **Automated Training Pipeline:** A multi-stage pipeline handles all steps from data acquisition to model registration.
- **Model Registry:** **AWS S3** is used as a centralized model registry to store, version, and retrieve the best-performing models.
- **Containerization:** **Docker** encapsulates the application and its dependencies, ensuring consistent behavior across different environments.
- **CI/CD Automation:** **GitHub Actions** automates the process of building, testing, and deploying the application upon code changes.
- **Cloud Deployment:** The containerized application is deployed and served from an **AWS EC2** instance, with container images managed by **AWS ECR**.

---

## 6. The Automated Training Pipeline

The heart of this project is the end-to-end training pipeline, which can be triggered on-demand (e.g., via a REST API endpoint). This pipeline automates every step of the model lifecycle.

**Pipeline Flow:**
Trigger --> [ Data Ingestion ] --> [ Data Validation ] --> [ Data Transformation ] --> [ Model Trainer ] --> [ Model Evaluation ] --> [ Model Pusher ] --> AWS S3
code
Code
- **Data Ingestion:** Fetches the dataset and prepares it for the pipeline.
- **Data Validation:** Validates incoming data against a predefined schema (`schema.yaml`) to ensure data quality and integrity, preventing pipeline failures due to unexpected data formats.
- **Data Transformation:** Applies a series of preprocessing steps identified during experimentation:
  - **Scaling:** `RobustScaler` to handle outliers.
  - **Imputation:** `SimpleImputer` with a 'median' strategy.
  - **Balancing:** `RandomUnderSampler` to address severe class imbalance.
- **Model Trainer:** Trains an `XGBoost` classifier on the transformed data.
- **Model Evaluation:** Compares the newly trained model against the current production model (if one exists) based on the primary business metric: **Total Cost**.
  - `Total Cost = (False Positives * $10) + (False Negatives * $500)`
- **Model Pusher:** If the new model shows a significant improvement (configurable threshold), it is versioned and pushed to the **AWS S3 model registry**, becoming the new production model.

---

## 7. CI/CD with GitHub Actions

The entire deployment process is automated using a CI/CD pipeline defined in `.github/workflows/aws.yaml`. This pipeline ensures that every push to the `main` branch results in a seamless and automated deployment of the latest version of the application.

**CI/CD Workflow:**
1.  **Trigger:** The workflow is automatically triggered on a `git push` to the `main` branch.
2.  **Self-Hosted Runner:** The job runs on a self-hosted runner configured on an **AWS EC2 instance**. This provides full control over the execution environment.
3.  **Build Docker Image:** A new Docker image for the application is built using the provided `Dockerfile`.
4.  **Push to ECR:** The newly built image is tagged and pushed to a private **AWS ECR (Elastic Container Registry)**.
5.  **Deploy on EC2:** The runner on the EC2 instance pulls the latest image from ECR and starts a new container, effectively deploying the updated application with zero downtime.

This hands-off deployment process ensures that the application is always up-to-date and consistently deployed.

---

# End-to-End MLOps: Predictive Maintenance for Scania APS

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)
![Platform](https://img.shields.io/badge/Platform-Docker-blue.svg)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub%20Actions-lightgrey.svg)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20ECR%20%7C%20EC2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains a complete, end-to-end MLOps solution for predicting Air Pressure System (APS) faults in Scania trucks. The project moves beyond a simple model in a notebook to a fully automated, production-ready system featuring a modular training pipeline, CI/CD automation, and cloud deployment.

---

## 8. Project Development & Deployment Workflow
This project was built following a systematic, end-to-end MLOps workflow, from initial setup to final automated deployment.

### Phase 1: Project Initialization & Environment Setup
> 🚀 Setting the foundation for a robust and reproducible project.

-   **Modular Scaffolding:** The project structure was automatically generated using a `template.py` script to ensure consistency and best practices from the start.
-   **Local Package Configuration:** `setup.py` was configured to make the entire `/src` directory an installable local package. This allows for clean, absolute imports and modular code across the project.
-   **Environment Setup:** A dedicated Conda environment was created using `Python 3.10`. All project dependencies were listed in `requirements.txt` and installed to ensure a fully reproducible environment.

### Phase 2: Experimentation & Core Utilities
> 🔬 From raw data to actionable insights and a resilient codebase.

-   **Rigorous Experimentation:** A detailed Jupyter Notebook (`Experimentation_notebook.ipynb`) was created to perform:
    -   Extensive Exploratory Data Analysis (EDA).
    -   Systematic testing of data preprocessing strategies (imputation, scaling, imbalance handling).
    -   Evaluation of multiple machine learning models to find the top performer.
    -   The insights from this notebook formed the blueprint for the final automated pipeline.
-   **Robust Logging & Exception Handling:** Centralized logging and custom exception handling systems were implemented to ensure the application is maintainable, debuggable, and resilient to errors.

### Phase 3: Building the Automated Training Pipeline
> ⛓️ Automating the core logic from data ingestion to a production-ready model.

The core of the project is a modular, multi-stage training pipeline. Each component was built systematically:
-   📥 **Data Ingestion:** Fetches the dataset and splits it into training and testing sets, outputting a `DataIngestionArtifact`.
-   ✅ **Data Validation:** Checks incoming data against a predefined `schema.yaml`. This step is crucial for maintaining data quality and preventing pipeline failures from data drift or corruption.
-   ✨ **Data Transformation:** Applies the optimal preprocessing steps discovered during experimentation (Robust Scaling, Median Imputation, Random Undersampling).

#### AWS Integration for Model Registry:
-   A dedicated **AWS S3 bucket** was created to act as a centralized and versioned Model Registry.
-   An **IAM User** with programmatic access was configured to allow the pipeline to securely push and pull models from the S3 bucket.

#### Model Training, Evaluation & Pushing:
-   🧠 **Model Trainer:** Trains the `XGBoost` model on the transformed data.
-   📊 **Model Evaluation:** Compares the newly trained model against the current production model (if any) based on our custom cost metric.
-   ⬆️ **Model Pusher:** Uploads the model to the S3 registry *only if* it meets a predefined performance threshold, making it the new production-ready model.

### Phase 4: Creating the Prediction Service
> 🌐 Serving the model and making it accessible to the end-user.

-   **Prediction Pipeline & Flask App:** A lightweight prediction pipeline was built to load the production model from S3 and perform inference. This was wrapped in a **Flask application** (`app.py`).
-   **User Interface:** Simple HTML templates and static files were created to provide a user-friendly interface for making predictions and triggering the training pipeline via API endpoints (`/predict`, `/train`).

### Phase 5: CI/CD Automation & Cloud Deployment
> 🚢 From a `git push` to a live application with zero manual intervention.

-   **Containerization:** A `Dockerfile` was written to containerize the Flask application, packaging it with all its dependencies for a portable and consistent deployment.
-   **AWS Infrastructure Setup:**
    -   **ECR (Elastic Container Registry):** A private repository was created to securely store the Docker images.
    -   **EC2 (Elastic Compute Cloud):** An Ubuntu server instance was launched to host the application, and Docker was installed on it.
-   **GitHub Actions & Self-Hosted Runner:**
    -   A CI/CD workflow was defined in `.github/workflows/aws.yaml`.
    -   The EC2 instance was configured as a **GitHub Self-Hosted Runner**, allowing GitHub Actions to execute deployment jobs directly on our server.
-   **Automated Deployment Workflow:**
    1.  **Secure Credentials:** `GitHub Secrets` were configured to securely store AWS credentials and the ECR repository URI.
    2.  **Trigger:** On every `git push` to the `main` branch, the GitHub Actions workflow automatically triggers.
    3.  **Build & Push:** The workflow builds the Docker image and pushes it to AWS ECR.
    4.  **Deploy:** The workflow then pulls the latest image on the EC2 runner and starts the new container, seamlessly updating the live application.
-   **Finalization:** The EC2 instance's security group was configured to allow inbound traffic on the required port, making the web application publicly accessible.

---

## 9. Technology Stack
- **Experimentation & Modeling:** Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Backend & Serving:** Flask
- **Cloud (AWS):** S3 (Model Registry), ECR (Container Registry), EC2 (Hosting)
- **DevOps:** Docker, GitHub Actions, Git
- **Environment Management:** Conda

---

## 10. Access the production (EC2) app

Run http://23.22.6.72:5000

**Important** : Use test data available in Application test data/Input data as input dictionary. Check the results using data/Output data corresponding to input file name.