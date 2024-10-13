import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Load the dataset from GitHub
@st.cache_data
def load_data():
    # Replace this URL with the raw GitHub URL of your CSV file
    url = r"https://github.com/Alko2122/Python-Group-Work/blob/609d9db50c0f9ddf5456d634e49ef9efb8fbe163/1553768847-housing.csv"
    response = requests.get(url)
    return pd.read_csv(StringIO(response.text))

df = load_data()
df_original = df.copy()

def clean_data(dataframe):
    cleaned_df = dataframe.copy()
    cleaned_df['total_bedrooms'] = cleaned_df['total_bedrooms'].fillna(cleaned_df['total_bedrooms'].median())
    return cleaned_df

def create_visualization(dataframe, columns, plot_type):
    if len(columns) == 0:
        st.warning("Please select at least one column for visualization.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    if plot_type == 'Histogram':
        for column in columns:
            sns.histplot(data=dataframe, x=column, kde=True, label=column, ax=ax)
        ax.legend()
        ax.set_title('Distribution of Selected Features')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    elif plot_type == 'Box Plot':
        sns.boxplot(data=dataframe[columns], ax=ax)
        ax.set_title('Box Plot of Selected Features')
        ax.set_ylabel('Value')

    elif plot_type == 'Scatter Plot':
        if len(columns) < 2:
            st.warning("Please select at least two columns for a scatter plot.")
            return
        sns.scatterplot(data=dataframe, x=columns[0], y=columns[1], ax=ax)
        ax.set_title(f'Scatter Plot: {columns[0]} vs {columns[1]}')

    elif plot_type == 'Violin Plot':
        sns.violinplot(data=dataframe[columns], ax=ax)
        ax.set_title('Violin Plot of Selected Features')
        ax.set_ylabel('Value')

    st.pyplot(fig)

    if len(columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(dataframe[columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap of Selected Features')
        st.pyplot(fig)

# Streamlit app
st.title('California House Price Data Analysis')

# Sidebar for user inputs
st.sidebar.header('User Input Features')

data_choice = st.sidebar.radio('Choose Data', ['Original Data', 'Cleaned Data'])

column_selector = st.sidebar.multiselect(
    'Select Columns for Visualization',
    options=df.columns.tolist(),
    default=[df.columns[0]]
)

plot_type = st.sidebar.selectbox(
    'Select Plot Type',
    options=['Histogram', 'Box Plot', 'Scatter Plot', 'Violin Plot']
)

# Main content
if data_choice == 'Original Data':
    current_df = df_original
else:
    current_df = clean_data(df.copy())

st.write(f"### Preview of {data_choice}")
st.write(current_df[column_selector].head())

if st.button('Visualize Data'):
    create_visualization(current_df, column_selector, plot_type)

# Display dataset info
st.sidebar.markdown("---")
st.sidebar.write("### Dataset Info")
st.sidebar.write(f"Total Records: {len(df)}")
st.sidebar.write(f"Total Features: {len(df.columns)}")
