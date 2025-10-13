# Business Case Study: Predictive Maintenance for Scania's Air Pressure System (APS)

### **1. Executive Summary**

Scania, a leader in the automotive industry, currently employs a manual, time-consuming, and cost-inefficient process for detecting critical faults in truck Air Pressure Systems (APS). This reactive approach leads to significant service costs and vehicle downtime. This document proposes the development of a predictive machine learning model that leverages existing sensor data from the vehicle's "black box" to proactively identify APS faults. The primary business driver is cost reduction, specifically by minimizing the high cost of missed detections (**False Negatives**), which are **50 times more expensive** than unnecessary checks (**False Positives**). The success of this initiative will be measured by a single, clear metric: the total cost incurred from prediction errors.

---

### **2. The Business Opportunity: From Reactive to Predictive Maintenance**

Scania's state-of-the-art trucks are equipped with numerous sensors that generate vast amounts of time-series data, tracking vehicle performance and health. This data is logged in the vehicle's black box and downloaded at service centers. While this data is rich with insights, its potential to pre-emptively identify mechanical failures remains largely untapped.

The current methodology for diagnosing faults in the Air Pressure System (APS)—a pivotal component of the truck's braking system—is a prime example of this inefficiency.

**The Core Problem:**
Every time a truck enters a service center, technicians must manually inspect the APS components to check for faults. This process is mandatory due to the critical safety nature of the braking system, but it is inherently flawed:
*   **Time-Consuming:** Manual inspection creates a significant bottleneck in the service workflow.
*   **Cost-Inefficient:** Labor and resources are spent inspecting healthy systems, adding unnecessary operational costs.
*   **Reactive, Not Proactive:** Faults are only discovered after they have potentially developed, not before.

> This project aims to transform this manual, reactive process into an automated, predictive one, saving both time and money while enhancing service quality.

---

### **3. The Proposed Solution: An ML-Powered Diagnostic Tool**

We will develop a predictive model that uses sensor readings collected at the service center to accurately classify whether a truck has an APS fault. By integrating this model into the service PC's diagnostic workflow, Scania can achieve a smarter, data-driven approach to maintenance.

**Data Foundation:**
The model will be trained on a robust dataset collected from over **36,000 Scania trucks**. This dataset has been pre-classified by domain experts into two distinct categories:
*   **`pos` (Positive Class):** The truck has a confirmed APS fault.
*   **`neg` (Negative Class):** The truck's APS is functioning correctly.

This high-quality dataset provides a strong foundation for building a model capable of predicting APS faults with high accuracy.

---

### **4. Financial Impact: The Business Cost of Prediction**

To build an effective model, we must align its performance with tangible business outcomes. The financial implications of our model's predictions are clear and quantifiable.

There are two primary costs associated with any prediction error:

*   **Cost 1: False Positive (`$10`)**
    *   **Scenario:** The model predicts an APS fault, but in reality, none exists.
    *   **Business Impact:** The service team performs an unnecessary manual inspection, incurring a nominal cost of **$10**.

*   **Cost 2: False Negative (`$500`)**
    *   **Scenario:** The model fails to detect a genuine APS fault.
    *   **Business Impact:** A truck with a faulty braking system leaves the service center. This could lead to critical failures on the road, resulting in extensive repairs, potential accidents, and damage to brand reputation. The estimated cost for such an event is **$500**.

> **Key Insight:** A False Negative is **50 times more costly and damaging** to the business than a False Positive. Therefore, the model must be aggressively optimized to minimize missed detections.

---

### **5. Success Metric: The Total Cost Formula**

The ultimate evaluation of this model will not be based on traditional accuracy metrics alone, but on a cost-centric formula that directly reflects its financial impact on the company.

The primary Key Performance Indicator (KPI) will be the **Total Cost of the Predictive Model**, calculated as follows:

Total Cost = (Number of False Positives × $10) + (Number of False Negatives × $500)

# 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

## Prerequisites

*   Python 3.10+
*   Conda or another virtual environment manager
*   An AWS account with configured IAM credentials

## Installation and Setup

1.  **Clone the Repository**
    Open your terminal and clone the project repository:
    ```bash
    git clone https://github.com/attri03/Full-Stack-Data-Science-APS.git
    cd Full-Stack-Data-Science-APS
    ```

2.  **Create and Activate a Virtual Environment**
    It is highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    # Create the conda environment
    conda create -n APSSensor python=3.10 -y

    # Activate the environment
    conda activate APSSensor
    ```

3.  **Configure Environment Variables**
    This project requires AWS credentials to access resources. Set the following environment variables in your terminal session.

    **For Windows (PowerShell):**
    ```powershell
    $env:AWS_REGION_NAME="your-aws-region"
    $env:AWS_SECRET_ACCESS_KEY="your-secret-access-key"
    $env:AWS_ACCESS_KEY_ID="your-access-key-id"
    ```
    **For macOS/Linux:**
    ```bash
    export AWS_REGION_NAME="your-aws-region"
    export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
    export AWS_ACCESS_KEY_ID="your-access-key-id"
    ```    
    > **Note:** For a more permanent and secure solution, consider using a `.env` file.

4.  **Install Required Dependencies**
    Install all the necessary packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Once the setup is complete, you can run the main application pipeline:

```bash
python demo.py
