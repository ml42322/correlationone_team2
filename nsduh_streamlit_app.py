import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# resources: https://docs.streamlit.io/knowledge-base/tutorials/databases/tableau
# https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace
# https://towardsdatascience.com/embedding-tableau-in-streamlit-a9ce290b932b
# https://docs.streamlit.io/library/get-started/installation#prerequisites


######### IMPORT DATA #########
@st.cache(suppress_st_warning=True)
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


######### RESOURCES #########
if section == "Resources":

    st.title('Resources')
