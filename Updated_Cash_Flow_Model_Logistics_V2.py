# Import all the library
import plotly.io as pio
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.tsa.api as smt
import statsmodels.api as sm
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import sys


def intial_clean_up():
    marketplace_df = pd.read_excel("Billing Status Sheet (Finance).xlsx",sheet_name="Aging")
    marketplace_df = marketplace_df[['Billed Date', 'Amount ', 'Amount Due',
                                     'Paid Amount', 'Due date', 'Payment / write-off date']]
    
    # drop rows where 'Billed Date' column is missing
    marketplace_df.dropna(subset=['Billed Date'], inplace=True)
    
    try:
        # convert date columns to datetime format
        date_cols = ['Billed Date', 'Due date', 'Payment / write-off date']
        marketplace_df[date_cols] = marketplace_df[date_cols].apply(
            pd.to_datetime, errors='coerce')

        # convert numeric columns to float format
        num_cols = ['Amount ', 'Amount Due', 'Paid Amount']
        marketplace_df[num_cols] = marketplace_df[num_cols].apply(
            pd.to_numeric, errors='coerce')

#         replace non-numeric values with NaN
        marketplace_df = marketplace_df.replace({pd.NaT: np.nan, None: np.nan})
        
    except Exception as e:
        # handle any exceptions that occur
        print(f"An error occurred while converting the data: {e}")

    return marketplace_df

    
def weekly_report(df, opening_balance):
    # Sort the DataFrame by Billed Date
    df_billed = df.copy()
    df_billed.sort_values(by='Billed Date', inplace=True)

    # Set the index to Billed Date
    df_billed.set_index('Billed Date', inplace=True)

    # Sort the DataFrame by Due date
    df_due = df.copy()
    df_due.sort_values(by='Due date', inplace=True)

    # Set the index to Due date
    df_due.set_index('Due date', inplace=True)

    # Sort the DataFrame by Payment Date
    df_paid = df.copy()
    df_paid.sort_values(by='Payment / write-off date', inplace=True)

    # Set the index to Payment Date
    df_paid.set_index('Payment / write-off date', inplace=True)

    # Define the start and end dates of the analysis period
    start_date = df_billed.index.min()
    end_date = df_billed.index.max()+timedelta(days=28)
#     print(start_date,end_date)

    # Create a list of weekly dates between the start and end dates
    weekly_dates = []
    current_date = start_date
    while current_date <= end_date:
        weekly_dates.append(current_date)
        current_date += timedelta(days=7)

    # Initialize the Amount Out, Amount Expected In, Amount Received, Aging Balance, and Closing Balance to empty lists
    amount_out = []
    amount_expected_in = []
    amount_received = []
    aging_balance = []
    closing_balance = []
    x=opening_balance

    # Iterate over the weekly dates
    for i in range(len(weekly_dates)):
        # Define the start and end dates of the current week
        if i == 0:
            current_start_date = start_date
        else:
            current_start_date = weekly_dates[i-1] + timedelta(days=1)
        current_end_date = weekly_dates[i]

        # Subset the DataFrame to the current week
        current_week_billed = df_billed.loc[(df_billed.index >= current_start_date) & (df_billed.index < current_end_date)]
        current_week_due = df_due.loc[(df_due.index >= current_start_date) & (df_due.index < current_end_date)]
        current_week_paid = df_paid.loc[(df_paid.index >= current_start_date) & (df_paid.index < current_end_date)]

        # Calculate the Amount Out, Amount Expected In, and Amount Received for the current week
        amount_out_current_week = current_week_billed['Amount '].sum()
        amount_expected_in_current_week = current_week_due['Amount Due'].sum()
        amount_received_current_week = current_week_paid['Paid Amount'].sum()
        aging_balance_current_week = amount_expected_in_current_week - amount_received_current_week

        # Append the Opening Balance, Amount Out, Amount Expected In, Amount Received, Aging Balance, and Closing Balance to the corresponding lists
        opening_balance -= aging_balance_current_week
        amount_out.append(-amount_out_current_week)
        amount_expected_in.append(amount_expected_in_current_week)
        amount_received.append(amount_received_current_week)
        aging_balance.append(-aging_balance_current_week)
        closing_balance.append(opening_balance)

    # Create a DataFrame from the lists and weekly_dates
    data = {'Opening Balance': [x] + closing_balance[:-1], 'Amount Out': amount_out, 'Amount Expected In': amount_expected_in, 'Amount Received': amount_received, 'Aging Balance': aging_balance, 'Closing Balance': closing_balance}
    weekly_df = pd.DataFrame(data, index=weekly_dates)
    return weekly_df

def handle_outliers_iqr(dataset):

# Cleans a dataset by removing or replacing outliers using the IQR method.
    # Calculate the IQR of the dataset
    Q1 = np.percentile(dataset, 25)
    Q3 = np.percentile(dataset, 75)
    IQR = Q3 - Q1
    
    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify the outliers
    outliers = dataset[(dataset < lower_bound) | (dataset > upper_bound)]
    
    # Remove the outliers
#     cleaned_dataset = dataset[(dataset >= lower_bound) & (dataset <= upper_bound)]
    
    # Replace the outliers with the median
    mean = dataset.mean()
    cleaned_dataset = np.where((dataset < lower_bound) | (dataset > upper_bound), mean, dataset)
    
    return cleaned_dataset

def forecasting_result(data):
    d, D, s, p, q, P, Q = 1, 1, 5, 4, 2, 0, 1
    best_model_amount_out=sm.tsa.statespace.SARIMAX(data['Amount Out'], order=(p, d, q),
                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)
    
    best_model_amount_in=sm.tsa.statespace.SARIMAX(data['Amount Received'], order=(p, d, q),
                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)
    
    forecast_amount_out = best_model_amount_out.forecast(steps=3)
    forecast_Amout_recieved = best_model_amount_in.forecast(steps=3)
    predicted_df=data.tail(3)
    predicted_df['Amount Out']=list(forecast_amount_out)
    predicted_df['Amount Received']=list(forecast_Amout_recieved)
    return predicted_df

def calculate_predicted_df(opening_balance, predicted_df):
    amount_out_1 = []
    amount_expected_in_1 = []
    amount_received_1 = []
    aging_balance_1 = []
    closing_balance_1 = []
    opening_balance_1 = opening_balance

    for index, row in predicted_df.iterrows():
        # Calculate the Amount Out, Amount Expected In, and Amount Received for the current week
        amount_out_pred = row['Amount Out']
        amount_expected_pred = row['Amount Expected In']
        amount_received_pred = row['Amount Received']
        aging_balance_pred = amount_expected_pred - amount_received_pred

        # Append the Opening Balance, Amount Out, Amount Expected In, Amount Received, Aging Balance, and Closing Balance to the corresponding lists
        opening_balance_1 -= aging_balance_pred
        amount_out_1.append(amount_out_pred)
        amount_expected_in_1.append(amount_expected_pred)
        amount_received_1.append(amount_received_pred)
        aging_balance_1.append(-aging_balance_pred)
        closing_balance_1.append(opening_balance_1)

    # Create a DataFrame from the lists and weekly_dates
    data_1 = {
        'Opening Balance': [opening_balance] + closing_balance_1[:-1], 
        'Amount Out': amount_out_1, 
        'Amount Expected In': amount_expected_in_1, 
        'Amount Received': amount_received_1, 
        'Aging Balance': aging_balance_1, 
        'Closing Balance': closing_balance_1
    }
    pred_next_three_df = pd.DataFrame(data_1, index=predicted_df.index)
    return pred_next_three_df

def run_cashflow_model(opening_balance):
    iter_1_df=intial_clean_up()
    iter_2_df=weekly_report(iter_1_df,opening_balance)
#     iter_2_df.to_excel('test_1.xlsx')
    cleaned_dataset_out = handle_outliers_iqr(iter_2_df['Amount Out'])
    cleaned_dataset_in = handle_outliers_iqr(iter_2_df['Amount Received'])
    trail_forecasted_df=iter_2_df.copy()
    trail_forecasted_df['Amount Out']=cleaned_dataset_out
    trail_forecasted_df['Amount Received']=cleaned_dataset_in
    forecasted_df=forecasting_result(trail_forecasted_df)
    opening_balance_updated=iter_2_df['Closing Balance'].iloc[-4]
    final_predicted_df=calculate_predicted_df(opening_balance_updated, forecasted_df)
    past_pred_df=iter_2_df.drop(index=iter_2_df.index[-3:])
    return past_pred_df,final_predicted_df


def plot_cashflow_graphs(initial_opening_balance):
    past_cashflow_df,future_cashflow_df=run_cashflow_model(initial_opening_balance)
    # Create the plotly figure object
    fig = go.Figure()
    # Add the traces
    fig.add_trace(go.Scatter(x=past_cashflow_df.index, y=past_cashflow_df['Closing Balance'],
                             mode='lines+markers', name='Closing Balance',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=past_cashflow_df.index, y=past_cashflow_df['Opening Balance'],
                             mode='lines+markers', name='Opening Balance',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=past_cashflow_df.index, y=past_cashflow_df['Amount Out'],
                             mode='lines+markers', name='Amount Out',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=past_cashflow_df.index, y=past_cashflow_df['Amount Expected In'],
                             mode='lines+markers', name='Amount Expected In',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=past_cashflow_df.index, y=past_cashflow_df['Amount Received'],
                             mode='lines+markers', name='Amount Received',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=past_cashflow_df.index, y=past_cashflow_df['Aging Balance'],
                             mode='lines+markers', name='Aging Balance',hoverlabel=dict(font_size=14,namelength=-1)))

    fig.add_trace(go.Scatter(x=future_cashflow_df.index, y=future_cashflow_df['Closing Balance'], 
                             mode='lines+markers', name='Forecasted Closing Balance',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=future_cashflow_df.index, y=future_cashflow_df['Opening Balance'], 
                             mode='lines+markers', name='Forecasted Opening Balance',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=future_cashflow_df.index, y=future_cashflow_df['Amount Out'], 
                             mode='lines+markers', name='Forecasted Amount Out',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=future_cashflow_df.index, y=future_cashflow_df['Amount Expected In'], 
                             mode='lines+markers', name='Forecasted Amount Expected In',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=future_cashflow_df.index, y=future_cashflow_df['Amount Received'], 
                             mode='lines+markers', name='Forecasted Amount Received',hoverlabel=dict(font_size=14,namelength=-1)))
    fig.add_trace(go.Scatter(x=future_cashflow_df.index, y=future_cashflow_df['Aging Balance'], 
                             mode='lines+markers', name='Forecasted Aging Balance',hoverlabel=dict(font_size=14,namelength=-1)))
    # Set the figure layout
    fig.update_layout(title='Balance Sheet', xaxis_title='Date', yaxis_title='Amount')
    fig.update_layout(
    title={'text':f'Balance Sheet (Initial Opening Balance : $ {initial_opening_balance:,})','x':0.3,'y':0.99},
    xaxis_title='Date',
    yaxis_title='Amount',
    yaxis=dict(
        tickprefix='$' ),hovermode='x',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))  # Add the dollar sign as a prefix to the tick values

    # Show the plot
    # pio.show(fig)
    pio.write_html(fig, 'my_plot.html')
    
    return future_cashflow_df
    

# Get the second command-line argument
argument = sys.argv[1]
plot_cashflow_graphs(float(argument))