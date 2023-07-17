Credit Default Capstone
==============================

Machine Learning project to predict credit default based on the customer's monthly profile.

- The dataset is from the [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction/overview)  kaggle competition. 
- Goal of the competition is predict credit default based on the customer's monthly profile.

## Project Organization

## Data Pipeline

![Data Pipeline](/Documentation/images/data-pipline%2007-12-2023.png)

### Data Pipeline Components
1. Raw data 
2. Simplelmputer 
3. Simplelmputer 
4. OrdinalEncoder 
5. OneHotEncoder
6. RemoveMulticollinearity 
7. FixImbalancer 
8. StandardScaler
9. SelectFromModel 
10. Cat BoostClassifier

## Data Storage

Git LFS is used to store the orginal dataset. The CSV file was converted to the feather format to reduce the size of the file. 

## ML Model Lifecycle Management

MLflow is used to track model training, model performance, and model deployment.


### Deployment Method

The project will be deployed using a combination of **Pycaret** and the **Gradio** library. 
* Pycaret will be used to train the model 
* Gradio will be used to create a web app that will allow users to interact with the model. 
* The web app will be hosted on **Azure**.




