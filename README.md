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

# Run Project

1. Clone the github repository
eg: git clone https://github.com/attri03/Full-Stack-Data-Science-APS.git

2. (Optional) Virtual environment in the terminal
A. Create virtual enviroment
eg: conda create -n APSSensor python=3.10 -y
B. Activate virtual environment in the terminal
eg: conda activate APSSensor

3. Use your environment vaiables:
$env:AWS_REGION_NAME=""
$env:AWS_SECRET_ACCESS_KEY="
$env:AWS_ACCESS_KEY_ID=""

4. Install requirements.txt
eg: pip install -r requirements.txt

5. Run demo.py in the terminal
eg: python demo.py