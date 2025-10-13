# Business Case Study

## Overview of the Scania Service station Anomaly detection mechanism
Scania, is a big Automotive which build state-of-the-art trucks for logistics. Scania truck has a lot of embedded technology for tracking the performance and detecting anomalies in the truck. There are many sensors in the truck. These sensors provide the time-series data to keep the track of the vehicle performance as well as detecting any anomaly in the truck. The key insights from the vast time-series data is stored in the black box of the vehicle. When the truck is bought to the service center, these key insights are sent to the service PC's through CAN from black-box. The service PC's run a lot of ML models in them for detecting anomalies in the system.

## APS fault through sensors and current methodology
One of the major fault that needs to be tackled by the service people is APS fault. APS plays a pivotal role in braking system's of the truck. It is crucial to find the faults in APS system and solve it before they go bad. Therefore, everytime the truck comes to the service center, service people open up the mechnical aspect and check wheter there was an issue with the APS or not. This process is time consuming and cost inefficient. Therefore, Scania needs a predictive model, which with the help of data that is stored in the black-box can predict whether there is APS fault issue or not. This method will solve both time-complexity as well as cost-efficiency.

## Data for predicting the APS Fault
Domain experts found out that, when the truck is bought to the service center, there are some readings in the sensor that directly or indirectly tells APS fault. These readings are nothing but the senor readings at the time when service of the vehicle is yet to start. They collected the data from 36000+ Scania trucks and classified the data into 2 classes `pos` and `neg`.

- `pos` : There is APS fault.
- `neg` : There is no APS fault.

This  data has the capability of predicting the APS fault with high accuracy.

## Business cost of the predictive model
There are primarily 2 costs associated with the predictive model. 
1. `cost 1` : If our model suggests that there is fault in the APS but in reality there was no fault. The total cost incurred by the company to check the APS sensor is 10 dollars. This is known as `False Positive`.
2. `cost 2` : If our model suggests that there is no fault in the APS but in reality there was a fault. The total cost incurred by the company is 500 dollars. This is known as `False Negative`.

So, `False Negative` is 50 times more harmful then `False Positive`. 

## Evaluation Metrics 
The total cost of the predictive model to the company will be given by:

`Total_cost = 10 * occurances_of_cost1 + 500 * occurances_of_cost2`

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
