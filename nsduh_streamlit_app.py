import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Colormap, LinearSegmentedColormap
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_option('deprecation.showPyplotGlobalUse', False)
		 

# resources: https://docs.streamlit.io/knowledge-base/tutorials/databases/tableau
# https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace
# https://towardsdatascience.com/embedding-tableau-in-streamlit-a9ce290b932b
# https://docs.streamlit.io/library/get-started/installation#prerequisites


######### IMPORT DATA #########
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def import_nsduh():
	nsduh = pd.read_csv('Data/nsduh_data_cleaned.csv')
	nsduh = nsduh[nsduh['PDEN10']!=3]
	nsduh.drop(['Unnamed: 0'], axis=1, inplace=True)
	return nsduh

nsduh = import_nsduh()

def import_hpsa():
	hpsa = pd.read_csv('Data/HPSA_Cleaned.csv')
	return hpsa

hpsa = import_hpsa()

def import_hrsa():
	hrsa = pd.read_csv('Data/grants_per_county_by_year.csv')
	return hrsa

hrsa = import_hrsa()

# figure out difference for local vs cloud


######### SIDE BAR #########

st.sidebar.markdown("""
## DS4A / Women 2021 - Team 2
# """)

st.sidebar.image('Images/logo.png')

st.sidebar.markdown("""
## Analyzing Susceptibility to Mental Health Issues
""")

# Outline Options for Sidebar
section = st.sidebar.selectbox("Navigation Bar", ("The Team", "Project Overview", "Datasets",
                               "Exploratory Data Analysis", "Methodology", "Findings and Recommendation"))

st.sidebar.markdown("""
### Team Members: 
Whitney Brooks (Executive) |  Catherine Greenman (Practitioner)  |  Margot Herman (Practitioner) |  Michell Li (Practitioner)  |  Chiu-Feng (Steph) Yap (Practitioner)
""")

######### ABOUT TEAM 2 #########
if section == "The Team":

	st.write("""

	### Project
	To learn more about how we analyzed the susceptibility of U.S. adults to mental health issues, check out the navigation bar on the left.

	### The Team
	##### [Chiu-Feng (Steph) Yap, MPH](https://linkedin.com/in/chiufengyap/)
	🧠  Data Analyst at a startup, MPH in Epidemiology with Advanced Data Science Certificate

	##### [Michell Li](https://www.linkedin.com/in/michell-li/)
	🧠  Business Intelligence Analyst at Instacart

	##### Margot Herman
	🧠  Degree in Computer Science, Software Engineer at a startup

	##### [Catherine Greenman](https://www.linkedin.com/in/csgreenman/)
	🧠  Data Engineer, Bachelor’s in Computer Science from Columbia University

	##### [Whitney Brooks](http://linkedin.com/in/hookedonbooks1930)
	🧠  SVP of Product Management at Wells Fargo
	""")

######### PROJECT OVERVIEW #########
if section == "Project Overview":

    st.title('Analyzing Susceptibility to Mental Health Issues')

    st.write('''
	## Problem Statement
	While the stigma around mental health has decreased over the years, many providers have seen a spike in cases related to “diseases of despair.” These include anxiety and depression, which often go untreated or lead sufferers to “self-medicate” with substances like drugs and alcohol. 
	
	According to the Tufts Medical Center and One Mind at Work, depression alone accounts for about $44 billion in losses to workplace productivity. In 2019, national spending on mental health services totaled $225.1 billion and accounted for 5.5% of all health spending. 
	
	Furthermore, approximately 40% of Americans live in a designated mental health provider shortage area, which exacerbates the problem. Across the US, each state has discretionary funding allocated specifically for mental health. Sufficient funds and effective resource allocation are necessary for the diagnosis and treatment of mental health issues. 	Adequate funding is a key factor in most leading implementation science frameworks. 
	
	When mental health issues left untreated, these challenges can have a negative effect on a person's economic solvency, leading to increased rates of homelessness and poverty, social isolation, deteriorating physical health and shorter life expectancy, and decreased profitability for employers and their shareholders due to lower employee efficiency. 

	
	## Objective

	Mental health issues are pervasive and, now more than ever, need to be better understood to address their causes and impacts in a meaningful way.

	The goal of this project is to identify factors that make individuals more susceptible to mental health issues, based on self-administered substance use, demographics, and geographic information from the National Survey on Drug Use and Health (NSDUH).
	
	## Problem Importance 
	If we could identify the type of population who is more likely to struggle from mental health issues and identify features/variables features that we could potentially use to provide insights in terms of where the funding for mental health services should be distributed.
	''')


######### DATASETS #########
if section == "Datasets":
	st.title('Datasets')
	st.markdown('''
	We leveraged three datasets from 2015-2019 for our statistical analysis:
	''')
	
	st.write('''
	### [National Survey on Drug Use and Health (NSDUH)](https://www.datafiles.samhsa.gov/dataset/national-survey-drug-use-and-health-2019-nsduh-2019-ds0001)
	This dataset contains survey level information on the trends in specific substance use and mental illness measures''')
	st.dataframe(nsduh.head(5))

	st.write('''
	### [Health Professional Shortage Areas (HPSA)](https://data.hrsa.gov/data/download#SHORT)
	This datasets contains clinic metadata, clinic geospatial data, and clinic-level data including date of designation and withdrawal''')
	st.dataframe(hpsa.head(5))

	st.write('''
	### [Health Resources and Services Administration (HRSA)](https://data.hrsa.gov/data/download)
	This dataset contains total funding and mental-health related funding on a state and county-level. ''')
	st.dataframe(hrsa.head(5))

	st.write('## Data Cleaning')

	st.write('''
	#### NSDUH
	The NSDUH dataset originally has 210, 959 records and more than 2000 columns. As the NSDUH consolidates location data into three overarching categories and does not preserve state or county-level data, we decided to consolidate five years’ worth of reports in order to use time as a metric for tracking trends in mental health indicators. The reports are yearly, from 2015 to 2019. 
	
	* Preserve only the columns relevant to our analysis (including answers to mental health screening questions, insurance status, and demographic data
	* Consolidate non-committal answers (don’t know, refused, legitimate skip) into a single value
	''')

	st.write('''
	#### HPSA
	The HPSA dataset has 27,813 rows and 65 columns. Rows are on the HPSA entity level. The following was done to clean the data:
	
	* Remove variables with more than 30% data missing
	* Remove variables without any variation (Break in Designation, Discipline Class Number, Data Warehouse Record Create Date Text)
	* The resulting dataset has 27,813 rows and 44 columns.''')

	st.write('''
	#### HRSA Awarded Grants and CBSA
	
	HRSA Awarded Grants data contains awards from 2013 through 2021. For the Exploratory Data Analysis, we just decided to look at grants from 2015 to 2019. 
	
	* Only keep columns related to Financial Assistance, Award Year,  Grant Program Description, and geographic information like State and County names 
	* Group by County and State and calculate the sum of the Financial Assistance as well as Financial Assistance related to mental health for all awards in a specific county from the HRSA dataset.
	* Inner join HRSA with the CBSA dataset on State and County in order to relate Financial Assistance for awarded grants per county with the county population
	''')
	st.write('## Feature Engineering')
	st.write('''
	#### NSDUH
	
	HRSA Awarded Grants data contains awards from 2013 through 2021. For the Exploratory Data Analysis, we just decided to look at grants from 2015 to 2019. The first goal of feature engineering for the HPSA dataset is to create a column that allows the dataset to be joined with the NSDUH dataset. The second goal is to generate features using existing features that could enhance the dataset. 
	
	* Created a column for population density based on CBSA population density definitions for 2010. (PDEN10)
	* Created Time spent as designated entity variable (DaysBeforeWithdrawn)
	''')

	st.markdown('''
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

	def show_plots():

		colors = sns.color_palette("cubehelix", n_colors=5)
		cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)

		st.write('''
		#### Figure 1. 
		Distribution of Susceptibility to Mental Health Issues by Population Density
		''')
		sns.countplot(y='Serious_Psychological_Distress_Indicator_Past_Month', hue='PDEN10', data=nsduh, palette='ch:start=.65,rot=-.5')
		st.pyplot()

		st.write('''
		#### Figure 2. 
		Distribution of Gender by Population Susceptibility to Mental Health Issues
		''')
		sns.countplot(y='Serious_Psychological_Distress_Indicator_Past_Month', hue='Gender', data=nsduh, palette='ch:start=.65,rot=-.5')
		st.pyplot()

		st.write('''
		#### Figure 3. 
		Proportion of Participants's Susceptibility to Mental Health Issues by Age Group
		''')
		table=pd.crosstab(nsduh.Age_Category_Six_Levels,nsduh.Serious_Psychological_Distress_Indicator_Past_Month)
		table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, colormap=cmap1)
		plt.xlabel('Race/Ethnicity')
		plt.ylabel('Proportion of Participants')
		st.pyplot()

		st.write('''
		#### Figure 4. 
		Proportion of Participants's Susceptibility to Mental Health Issues by Race/Ethnicty
		''')
		table=pd.crosstab(nsduh.Race_Ethnicity,nsduh.Serious_Psychological_Distress_Indicator_Past_Month)
		table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, colormap=cmap1)
		plt.xlabel('Race/Ethnicity')
		plt.ylabel('Proportion of Participants')
		st.pyplot()

		st.write('''
		#### Figure 5.
		[Interactive Infographic on Tableau](https://public.tableau.com/app/profile/chiu.feng.yap/viz/DS4A-DataDivasFinalProjectDeliverable/Infographic)
		''')
		st.image('DS4A_Team2_Datafolio.jpg')

	show_plots()


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


