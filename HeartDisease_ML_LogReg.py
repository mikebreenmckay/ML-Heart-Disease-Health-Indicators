import pandas as pd

df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")
strong_corr = ['HeartDiseaseorAttack','HighBP', 'HighChol', 'Smoker',
               'Stroke', 'Diabetes', 'GenHlth', 'PhysHlth','MentHlth',
               'DiffWalk', 'Age', 'Education', 'Income']
df = df[strong_corr]
