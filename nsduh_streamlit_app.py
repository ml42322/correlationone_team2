import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## resources: https://docs.streamlit.io/knowledge-base/tutorials/databases/tableau 
## https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace
## https://towardsdatascience.com/embedding-tableau-in-streamlit-a9ce290b932b
## https://docs.streamlit.io/library/get-started/installation#prerequisites


######### IMPORT DATA #########
@st.cache(suppress_st_warning=True)
def import_data():
	nsduh = pd.read_csv('./Data/nsduh_data_cleaned.csv')


# figure out difference for local vs cloud 


######### SIDE BAR #########
st.sidebar.markdown("""
# Analyzing Susceptibility to Mental Health Issues
""")
#### Outline Options for Sidebar
section = st.sidebar.selectbox("Outline",("Executive Summary","Datasets","Exploratory Data Analysis","Methodology","Findings and Recommendation","Resources"))

st.sidebar.markdown("""
### DS4A Women 2021 - Team 2
#### Whitney Brooks (Executive)
#### Catherine Greenman (Practitioner) 
#### Margot Herman (Practitioner)
#### Michell Li (Practitioner)
#### Chiu-Feng (Steph) Yap (Practitioner)
""")



######### EXECUTIVE SUMMARY #########
if section == "Executive Summary":

	st.title('Analyzing Susceptibility to Mental Health Issues')

	st.write('Executive Summary')


######### DATASETS #########
if section == "Datasets":

	st.title('Datasets')



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









