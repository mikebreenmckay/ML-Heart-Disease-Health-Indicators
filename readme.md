## Heart Disease Health Indicators Machine Learning Project

#### Author: Michael Breen-McKay
#### Date: 03/15/2023
---
##### 1. Project Goals: 
* Perform EDA on the data sourced from Kaggle.
* Create various machine learning models and check their performance.
* Create some nice data visualizations along the way.
* Explain my process in detail.

##### 2. The Data:
* Data was posted to Kaggle by Alex Teboul and listed as '253,680 survey responses from cleaned BRFSS 2015 - binary classification'
* While this data is already cleaned, I am interested in pulling raw data from the CDC for more recent years for the purpose of cleaning the raw data and having additional data to test/train the model on.
* As a voluntary survey we will need to be particularly aware of biases in the data.  This has been cleaned of null values so it will be important to check the distributions of the values.

###### 2a. Description of indicator/features from the survey:
* Response Variable / Dependent Variable:
    * Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) --> HeartDiseaseorAttack
* Independent Variables:
    * High Blood Pressure
        * Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional --> HighBP
    * High Cholesterol
        * Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high? --> HighChol
        * Cholesterol check within past five years --> CholCheck
    * BMI
        * Body Mass Index (BMI) --> BMI
    * Smoking
        * Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] --> Smoker
    * Other Chronic Health Conditions
        * (Ever told) you had a stroke. --> Stroke
        * (Ever told) you have diabetes (If "Yes" and respondent is female, ask "Was this only when you were pregnant?". If Respondent says pre-diabetes or borderline diabetes, use response code 4.) --> Diabetes
    * Physical Activity
        * Adults who reported doing physical activity or exercise during the past 30 days other than their regular job --> PhysActivity
    * Diet
        * Consume Fruit 1 or more times per day --> Fruits
        * Consume Vegetables 1 or more times per day --> Veggies
    * Alcohol Consumption
        * Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) --> HvyAlcoholConsump
    * Health Care
        * Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service? --> AnyHealthcare
        * Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? --> NoDocbcCost
    * Health General and Mental Health
        * Would you say that in general your health is: --> GenHlth
        * Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? --> MentHlth
        * Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? --> PhysHealth
        * Do you have serious difficulty walking or climbing stairs? --> DiffWalk
    * Demographics
        * Indicate sex of respondent. --> Sex
        * Fourteen-level age category --> Age
        * What is the highest grade or year of school you completed? --> Education
        * Is your annual household income from all sources: (If respondent refuses at any income level, code "Refused.") --> Income

