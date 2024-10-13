import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Load the dataset from GitHub
def load_data():
    # Use the raw GitHub URL of your CSV file
    url = "https://raw.githubusercontent.com/Alko2122/Python-Group-Work/refs/heads/main/1553768847-housing.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return None

    content = response.text
    
    try:
        df = pd.read_csv(StringIO(content))
        st.write(f"First few lines: {df.shape}")
        return df
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while parsing CSV: {str(e)}")
        return None

def clean_data(dataframe):
    cleaned_df = dataframe.copy()
    # Replace missing values with median for numerical columns
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # For categorical columns, fill with mode
    categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    
    return cleaned_df

def create_visualization(dataframe, columns, plot_type):
    if len(columns) == 0:
        st.warning("Please select at least one column for visualization.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == 'Histogram':
        for column in columns:
            sns.histplot(data=dataframe, x=column, kde=True, ax=ax)
        ax.set_title('Histogram of Selected Features')
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
        corr_fig, corr_ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(dataframe[columns].corr(), annot=True, cmap='coolwarm', ax=corr_ax)
        corr_ax.set_title('Correlation Heatmap of Selected Features')
        st.pyplot(corr_fig)

# Streamlit app
st.title('California House Price Data Analysis')

# Load data
df = load_data()

if df is not None and not df.empty:
    df_original = df.copy()
    st.write("Data loaded successfully. Here's a preview:")
    st.write(df.head())

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

    if st.button('Visualize Data'):
        create_visualization(current_df, column_selector, plot_type)

    # Display dataset info
    st.sidebar.markdown("---")
    st.sidebar.write("### Dataset Info")
    st.sidebar.write(f"Total Records: {len(df)}")
    st.sidebar.write(f"Total Features: {len(df.columns)}")

else:
    st.error("Failed to load data or the dataset is empty. Please check the error messages above and ensure the GitHub URL is correct.")
    st.stop()
