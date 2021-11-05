import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
		 

# resources: https://docs.streamlit.io/knowledge-base/tutorials/databases/tableau
# https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace
# https://towardsdatascience.com/embedding-tableau-in-streamlit-a9ce290b932b
# https://docs.streamlit.io/library/get-started/installation#prerequisites


######### IMPORT DATA #########
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def import_data():
    nsduh = pd.read_csv('./Data/nsduh_data_cleaned.csv')
    return nsduh


df = import_data()

# figure out difference for local vs cloud


######### SIDE BAR #########
st.sidebar.markdown("""
# Analyzing Susceptibility to Mental Health Issues
""")
# Outline Options for Sidebar
section = st.sidebar.selectbox("Outline", ("Executive Summary", "Datasets",
                               "Exploratory Data Analysis", "Methodology", "Findings and Recommendation", "Resources"))

st.sidebar.markdown("""
# DS4A Women 2021 - Team 2
# Whitney Brooks (Executive)
# Catherine Greenman (Practitioner)
# Margot Herman (Practitioner)
# Michell Li (Practitioner)
# Chiu-Feng (Steph) Yap (Practitioner)
""")


######### EXECUTIVE SUMMARY #########
if section == "Executive Summary":

    st.title('Analyzing Susceptibility to Mental Health Issues')
    st.write('''
	## Problem Statement
	While the stigma around mental health has decreased over the years, many providers have seen a spike in cases related to “diseases of despair.” These include anxiety and depression, which often go untreated or lead sufferers to “self-medicate” with substances like drugs and alcohol. According to the Tufts Medical Center and One Mind at Work, depression alone accounts for about $44 billion in losses to workplace productivity. In 2019, national spending on mental health services totaled $225.1 billion and accounted for 5.5% of all health spending1. Furthermore, approximately 40% of Americans live in a designated mental health provider shortage area, which exacerbates the problem. Across the US, each state has discretionary funding allocated specifically for mental health. Sufficient funds and effective resource allocation are necessary for the diagnosis and treatment of mental health issues. Mental health issues are pervasive and, now more than ever, need to be better understood to address their causes and impacts in a meaningful way.

	## Objective
	The goal of this project is to identify factors that make individuals more susceptible to mental health issues, based on self-administered substance use, demographics, and geographic information from the National Survey on Drug Use and Health (NSDUH).
	''')


######### DATASETS #########
if section == "Datasets":

    st.title('Datasets')
    st.write(''' ### Top 10 rows of NSDUH dataset ''')
    st.dataframe(df.head(10))
    st.write('''
	## Outcome Variable
	The outcome variable is captured in the data as a binary indicator of 1 (Yes) for ‘Past Month Serious Psychological Distress Indicator’ which is derived from a series of six questions, asking adults respondents how frequently they experienced the following symptoms in the past 30 days:

		* How often did you feel nervous?
		* How often did you feel hopeless?
		* How often did you feel restless or fidgety?
		* How often did you feel so sad/depressed that nothing could cheer you up?
		* How often did you feel that everything was an effort?
		* How often did you feel down on yourself, no good or worthless?

	Questions are asked on a likert scale of 1-5, with a sum greater than 13 being the threshold for the outcome variable.
	''')


######### EXPLORATORY DATA ANALYSIS #########
if section == "Exploratory Data Analysis":

    st.title('Exploratory Data Analysis')


######### METHODOLOGY #########
if section == "Methodology":

    st.title('Methodology')


######### FINDINGS AND RECOMMENDATION #########
if section == "Findings and Recommendation":

    st.title('Findings and Recommendation')
    cols = ['Overall_Health_Fair_Poor',
			'Education_Category_Less_than_HS',
			'Perceived_Unmet_Need',
			'Adult_Employment_Status_Unemployed',
			'Education_Category_HS_Grad',
			'Overall_Health_Good',
			'Worst_Psychological_Distress_Level',
			'Adult_Employment_Status_Other',
			'Education_Category_Some_College_Assoc',
			'Gender_Male',
			'Num_Days_Skipped_Work_Past_30_Days',
			'Year',
			'Total_Income_Family_Recode_75000orMore',
			'Age_Category_Six_Levels_50-64',
			'Age_Category_Six_Levels_35-49',
			'Age_Category_Six_Levels_26-34',
			'Age_Category_Six_Levels_65_And_Above',
			'Overall_Health_Very_Good',
			'Race_Ethnicity_Black',
			'Treatment_Type_Past_Year_Inpatient_Only',
			'PDEN10_Less_than_1_Mil']

    df['Overall_Health_Fair_Poor'] = np.where(df['Overall_Health'].isin([4,5]),1,0)
    df['Education_Category_Less_than_HS'] = np.where(df['Education_Category'] == 1,1,0)
    df['Adult_Employment_Status_Unemployed'] = np.where(df['Adult_Employment_Status'] == 3,1,0)
    df['Education_Category_HS_Grad'] = np.where(df['Education_Category'] == 2,1,0)
    df['Overall_Health_Good'] = np.where(df['Overall_Health'] == 3,1,0)
    df['Adult_Employment_Status_Other'] = np.where(df['Adult_Employment_Status'] == 4,1,0)
    df['Education_Category_Some_College_Assoc'] = np.where(df['Education_Category'] == 3,1,0)
    df['Gender_Male'] = np.where(df['Gender'] == 1, 1,0)
    df['Total_Income_Family_Recode_75000orMore'] = np.where(df['Total_Income_Family_Recode'] == 4,1,0)
    df['Age_Category_Six_Levels_50-64'] = np.where(df['Age_Category_Six_Levels'] == 5,1,0)
    df['Age_Category_Six_Levels_35-49'] = np.where(df['Age_Category_Six_Levels'] == 4,1,0)
    df['Age_Category_Six_Levels_26-34'] = np.where(df['Age_Category_Six_Levels'] == 3,1,0)
    df['Age_Category_Six_Levels_65_And_Above'] = np.where(df['Age_Category_Six_Levels'] == 6,1,0)
    df['Overall_Health_Very_Good'] = np.where(df['Overall_Health'] == 2,1,0)
    df['Race_Ethnicity_Black'] = np.where(df['Race_Ethnicity'] == 2,1,0)
    df['Treatment_Type_Past_Year_Inpatient_Only'] = np.where(df['Treatment_Type_Past_Year'] == 1,1,0)
    df['PDEN10_Less_than_1_Mil'] = np.where(df['PDEN10'] == 2,1,0)
    df = df.dropna()
    X = df.drop(['Serious_Psychological_Distress_Indicator_Past_Month','Id','Serious_Psychological_Distress_Indicator_Past_Year'], axis=1) # independent variables data
    y = df['Serious_Psychological_Distress_Indicator_Past_Month']  # dependent variable data
    

    st.write(''' ### Model Output ''')
    st.echo()
    with st.echo():
    	CX_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.33, random_state=42)
    	logisticRegr = LogisticRegression()
    	logisticRegr.fit(X_test, y_test)
    	predictions = logisticRegr.predict(X_test)
    	score = logisticRegr.score(X_test, y_test)
    	st.write('Accuracy for X_forward: {}'.format(score))

    	res = sm.Logit(y, X[cols]).fit()
    	st.write(res.summary())    	

    st.write(''' ### Model Accuracy ''')
    with st.echo():
    	yhat = logisticRegr.predict(X_test)
    	prediction = list(map(round, yhat))

    	# confusion matrix
    	cm = confusion_matrix(y_test, prediction)
    	st.write("Confusion Matrix : \n", cm)

    	# accuracy score of the model
    	st.write('Test accuracy = ', accuracy_score(y_test, prediction))


    st.write(''' ### Adjusted Odds Ratio ''')

    with st.echo():
    	params = res.params
    	conf = res.conf_int()
    	conf['Odds Ratio'] = params
    	conf.columns = ['5%', '95%','Odds Ratio']
    	st.write(np.exp(conf))




######### RESOURCES #########
if section == "Resources":

    st.title('Resources')
