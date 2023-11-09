import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import plotly.subplots as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.offline as pyo

# Load your dataset
data = pd.read_csv('./bike-sharing_hourly.csv')

# Data Preprocessing
data.rename(columns={'instant':'rec_id','dteday':'datetime','yr':'year','mnth':'month','weathersit':'weather_condition',
                   'hum':'humidity','cnt':'total_count'},inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data['season'] = data['season'].astype('category')
data['year'] = data['year'].astype('category')
data['month'] = data['month'].astype('category')
data['holiday'] = data['holiday'].astype('category')
data['weekday'] = data['weekday'].astype('category')
data['workingday'] = data['workingday'].astype('category')
data['weather_condition'] = data['weather_condition'].astype('category')

def remove_outliers(df, start_column_index, end_column_index, z_threshold=3):
    df_no_outliers = df.copy()
    
    for col in df_no_outliers.columns[start_column_index:end_column_index]:
        z_scores = np.abs((df_no_outliers[col] - df_no_outliers[col].mean()) / df_no_outliers[col].std())
        df_no_outliers = df_no_outliers[z_scores <= z_threshold]
    
    return df_no_outliers

filtered_data = remove_outliers(data, start_column_index=13, end_column_index = 17)

def calculate_daylight_hours(row):
    if 0 <= row['hr'] <= 7:
        return 0  # Night
    elif 8 <= row['hr'] <= 17:
        return 1  # Day
    else:
        return 0  # Night

filtered_data['daylight_hours'] = filtered_data.apply(calculate_daylight_hours, axis=1)
filtered_data['lag_1'] = filtered_data['total_count'].shift(1)
filtered_data['lag_24'] = filtered_data['total_count'].shift(24)
filtered_data=filtered_data.iloc[24:]
model_data = filtered_data.drop(['rec_id', 'datetime', 'season', 'atemp', 'registered', 'casual'], axis=1)

st.set_option('deprecation.showPyplotGlobalUse', False)
profile = ProfileReport(data)

# Title and description
st.title('Predicting total number of bikes')

st.sidebar.header('Introduction')
if st.sidebar.checkbox('Problem statement'):
    # Goal 1
    st.markdown("<p style='color:#FF5733;font-size:16px;'>1. Deep Analysis</p>", unsafe_allow_html=True)
    st.write("Understand how citizens use the bike-sharing service for cost optimization and service improvements.")

    # Goal 2
    st.markdown("<p style='color:#FF5733;font-size:16px;'>2. Predictive Model</p>", unsafe_allow_html=True)
    st.write("Develop a model to forecast hourly bicycle users, optimizing bike provisioning and reducing outsourcing costs.")

    # Goal 3
    st.markdown("<p style='color:#FF5733;font-size:16px;'>3. Integration</p>", unsafe_allow_html=True)
    st.write("Integrate analysis and predictive model to make informed decisions for cost optimization and service enhancement.")

# Sidebar for customization
st.sidebar.header('EDA steps')

if st.sidebar.checkbox('Raw Data'):
    st.subheader('Raw Data')
    st.write(data)

# Summary Statistics
if st.sidebar.checkbox('Summary Statistics'):
    st.subheader('Summary Statistics')
    st.write(data.describe())
    st.write("In addition, the data has 0 null values")

if st.sidebar.checkbox('Outlier analysis'):
    st.subheader('Outlier analysis')
    st.write('The barplots below show each variable could have some outliers')
    selected_columns = data.columns[11:]

    # Create subplots for the first set of box plots with a wider figure
    fig1 = sp.make_subplots(
        rows=1, 
        cols=len(selected_columns),
        shared_yaxes=False,
        column_widths=[0.6] * len(selected_columns),  # Adjust the width as needed
    )

    # Define the figure size with increased height and width
    fig1.update_layout(
        title='Boxplots for Unfiltered Variables',
        width=1000,  # Adjust the width as needed
        height=400,
        showlegend = False
    )

    for i, col in enumerate(selected_columns):
        box_trace = go.Box(y=data[col], name=f'Boxplot for {col}')
        fig1.add_trace(box_trace, row=1, col=i+1)

    # Update layout for the y-axis title
    fig1.update_yaxes(title_text='Values', row=1, col=1)

    st.plotly_chart(fig1)
    st.write('We made a function to remove outliers with a z-threshold of 3')

    fig2 = sp.make_subplots(
        rows=1, 
        cols=len(selected_columns),
        shared_yaxes=False,
        column_widths=[0.6] * len(selected_columns),  # Adjust the width as needed
    )

    # Define the figure size with increased height and width
    fig2.update_layout(
        title='Boxplots for Filtered Variables',
        width=1000,  # Adjust the width as needed
        height=400,
        showlegend = False
    )

    for i, col in enumerate(selected_columns):
        box_trace = go.Box(y=filtered_data[col], name=f'Boxplot for {col}')
        fig2.add_trace(box_trace, row=1, col=i+1)

    # Update layout for the y-axis title
    fig2.update_yaxes(title_text='Values', row=1, col=1)

    st.plotly_chart(fig2)

if st.sidebar.checkbox('Feature engineering'):
    st.subheader('Feature Engineering')
    st.write("We created 3 new variables from the data: daylight_hours, lag_1 and lag_24.")
    st.write("- daylight_hours': If the hours are between 8 and 17, then it returns 1 for day. Otherwise it returns the value 0 (for night)")
    st.write("- lag_1': The number of bikes rented for the previous hour")
    st.write("- lag_24': The number of bikes rented 24 hours earlier")

    # Create a histogram
    fig = px.histogram(filtered_data, x='daylight_hours', nbins=2, category_orders={'x': ['0', '1']})
    fig.update_xaxes(tickvals=[0, 1], ticktext=['Night', 'Day'])
    fig.update_xaxes(title='Daylight Hours')
    fig.update_yaxes(title='Frequency')
    fig.update_layout(title='Distribution of Daylight Hours')
    st.plotly_chart(fig)

    st.write("As we can see the counts are rather evenly distributed")

    from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(data['total_count'], lags=30, ax=ax)
    plt.title('Partial Autocorrelation Function (PACF) of Bike Usage')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.grid(True)
    plt.show()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(data['total_count'], lags=30, ax=ax)
    plt.title('Autocorrelation Function (ACF) of Bike Usage')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.grid(True)
    plt.show()
    st.pyplot(fig)

    st.write("As we can see the 2 lags we selected show significant autocorrelation for both the ACF and PACF")

if st.sidebar.checkbox('Graphical analysis'):
    st.subheader('Graphical analysis')
    import plotly.graph_objects as go
    import pandas as pd

    # Assuming you have a DataFrame 'df' with the necessary data

    # Create subplots with 1 row and 3 columns
    fig = go.Figure()

    # Plot the relationship between weekday and total users
    fig.add_trace(
        go.Box(
            x=data['weekday'],
            y=data['total_count'],
            name='Total Users',
            marker_color='blue',
            boxpoints='outliers'
        )
    )

    # Plot the relationship between weekday and casual users
    fig.add_trace(
        go.Box(
            x=data['weekday'],
            y=data['casual'],
            name='Casual Users',
            marker_color='green',
            boxpoints='outliers'
        )
    )

    # Plot the relationship between weekday and registered users
    fig.add_trace(
        go.Box(
            x=data['weekday'],
            y=data['registered'],
            name='Registered Users',
            marker_color='orange',
            boxpoints='outliers'
        )
    )

    # Update the layout
    fig.update_layout(
        title='Relationship between Weekday and User Types',
        xaxis_title='Weekday',
        yaxis_title='User Count',
        template='plotly'
    )

    # Show the plot
    st.plotly_chart(fig)

    # Pivot tables
    # Create pivot tables for each user type
    pivot_total = data.pivot_table(values='total_count', index='weekday', columns='hr', aggfunc='sum')
    pivot_registered = data.pivot_table(values='registered', index='weekday', columns='hr', aggfunc='sum')
    pivot_casual = data.pivot_table(values='casual', index='weekday', columns='hr', aggfunc='sum')

    import plotly.graph_objects as go

    # Create a heatmap for Hourly Total Users by Day of the Week
    fig_total = go.Figure(data=go.Heatmap(
        z=pivot_total.values,
        x=pivot_total.columns,
        y=pivot_total.index,
        colorscale='plasma',
        zmin=0,
        zmax=pivot_total.values.max(),
        colorbar=dict(title='Total Users')
    ))
    fig_total.update_layout(
        title='Hourly Total Users by Day of the Week',
        xaxis_title='Hour of the Day',
        yaxis_title='Day of the Week',
        template='plotly'
    )

    # Create a heatmap for Hourly Registered Users by Day of the Week
    fig_registered = go.Figure(data=go.Heatmap(
        z=pivot_registered.values,
        x=pivot_registered.columns,
        y=pivot_registered.index,
        colorscale='plasma',
        zmin=0,
        zmax=pivot_registered.values.max(),
        colorbar=dict(title='Registered Users')
    ))
    fig_registered.update_layout(
        title='Hourly Registered Users by Day of the Week',
        xaxis_title='Hour of the Day',
        yaxis_title='Day of the Week',
        template='plotly'
    )

    # Create a heatmap for Hourly Casual Users by Day of the Week
    fig_casual = go.Figure(data=go.Heatmap(
        z=pivot_casual.values,
        x=pivot_casual.columns,
        y=pivot_casual.index,
        colorscale='plasma',
        zmin=0,
        zmax=pivot_casual.values.max(),
        colorbar=dict(title='Casual Users')
    ))
    fig_casual.update_layout(
        title='Hourly Casual Users by Day of the Week',
        xaxis_title='Hour of the Day',
        yaxis_title='Day of the Week',
        template='plotly'
    )

    # Show the heatmaps
    st.plotly_chart(fig_total)
    st.plotly_chart(fig_registered)
    st.plotly_chart(fig_casual)

    st.write("The graphs above show that registered and casual users clearly have different behaviors.")

    # Aggregate the data by month and weekday
    agg_data = data.groupby(['month', 'season'])['total_count'].mean().reset_index()

    # Create a bar plot for Weekday Wise Monthly Distribution of Counts using Plotly
    fig_month = px.bar(agg_data, x='month', y='total_count', color='season',
                          labels={'month': 'Month', 'total_count': 'Total Count'},
                          title='Season Wise Monthly Distribution of Counts (Plotly Bar Plot)',
                          category_orders={'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})  # Specify category order for months

    # Customize the x-axis labels to display the actual months (e.g., 1 to 'January', 2 to 'February', etc.)
    fig_month.update_xaxes(type='category', categoryorder='array',
                             categoryarray=['January', 'February', 'March', 'April', 'May', 'June',
                                            'July', 'August', 'September', 'October', 'November', 'December'])

    # Show the chart
    st.plotly_chart(fig_month)

    # Aggregate the data by month and weekday
    agg_data = data.groupby(['month', 'weekday'])['total_count'].mean().reset_index()

    # Create a bar plot for Weekday Wise Monthly Distribution of Counts using Plotly
    fig_weekday = px.bar(agg_data, x='month', y='total_count', color='weekday',
                          labels={'month': 'Month', 'total_count': 'Total Count'},
                          title='Weekday Wise Monthly Distribution of Counts (Plotly Bar Plot)',
                          category_orders={'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})  # Specify category order for months

    # Customize the x-axis labels to display the actual months (e.g., 1 to 'January', 2 to 'February', etc.)
    fig_weekday.update_xaxes(type='category', categoryorder='array',
                             categoryarray=['January', 'February', 'March', 'April', 'May', 'June',
                                            'July', 'August', 'September', 'October', 'November', 'December'])

    # Show the chart
    st.plotly_chart(fig_weekday)

    st.write('There is no significant change in a specific weekday based on the month.')

    fig_violin = go.Figure()
    fig_violin.add_trace(
        go.Violin(
            x=data['year'],
            y=data['total_count'],
            box_visible=True,
            line_color='blue',
            meanline_visible=True,
            name='Total Count'
        )
    )
    fig_violin.update_layout(
        title='Yearly Distribution of Counts (Violin Plot)',
        xaxis_title='Year',
        yaxis_title='Total Count',
        template='plotly'
    )

    # Show the violin plot
    st.plotly_chart(fig_violin)

    st.write("The bike usage when compared by year is clearly different. One notable difference is that 2012 was a better year for the company. However, the data is clearly more scattered.")

    avg_data = data.groupby(['season', 'holiday'])['total_count'].mean().reset_index()

    # Create a bar plot using Plotly
    fig_holiday = px.bar(avg_data, x='holiday', y='total_count', color='season',
                            labels={'season': 'Season', 'total_count': 'Total Count'},
                            title='Holiday Wise Distribution of Counts (Average)')

    # Customize the x-axis labels to display the actual season names
    fig_holiday.update_xaxes(type='category', categoryorder='array',
                                categoryarray=['Spring', 'Summer', 'Fall'])

    # Show the chart
    st.plotly_chart(fig_holiday)

    st.write("Users seem to use there bikes less on holidays.")

    avg_data = data.groupby(['season', 'workingday'])['total_count'].mean().reset_index()

    # Create a bar plot using Plotly
    fig_workingday = px.bar(avg_data, x='workingday', y='total_count', color='season',
                            labels={'season': 'Season', 'total_count': 'Total Count'},
                            title='Workingday Wise Distribution of Counts (Average)')

    # Customize the x-axis labels to display the actual season names
    fig_workingday.update_xaxes(type='category', categoryorder='array',
                                categoryarray=['Spring', 'Summer', 'Fall'])

    st.plotly_chart(fig_workingday)

    st.write("The assumption above is confirmed by this graph as the count is higher for workingdays than not.")

    # Create scatterplot for temperature vs. total_count with regression line
    # Create scatterplot for temperature vs. total_count with red regression line
    fig_temp = px.scatter(data, x='temp', y='total_count',
                          labels={'temp': 'Temperature', 'total_count': 'Total Count'},
                          title='Relation between temperature and users (Plotly Scatter Plot with Red Regression Line)',
                          trendline='ols',
                          trendline_color_override='red')  # Add red regression line

    # Create scatterplot for humidity vs. total_count with red regression line
    fig_humidity = px.scatter(data, x='humidity', y='total_count',
                              labels={'humidity': 'Humidity', 'total_count': 'Total Count'},
                              title='Relation between humidity and users (Plotly Scatter Plot with Red Regression Line)',
                              trendline='ols',
                              trendline_color_override='red')

    # Create scatterplot for windspeed vs. total_count with red regression line
    fig_windspeed = px.scatter(data, x='windspeed', y='total_count',
                               labels={'windspeed': 'Windspeed', 'total_count': 'Total Count'},
                               title='Relation between windspeed and users (Plotly Scatter Plot with Red Regression Line)',
                               trendline='ols',
                               trendline_color_override='red')

    
    # Display the scatterplots with red regression lines directly within Streamlit
    st.plotly_chart(fig_temp)
    st.write("The more the temperature increases the more users there are.")

    st.plotly_chart(fig_humidity)
    st.write("Humidity is negatively related to total count")
    
    st.plotly_chart(fig_windspeed)
    st.write("Windspeed seems to have a slightly positive relationship with total count.")
    
if st.sidebar.checkbox('Variable selection'):
    st.subheader('Variable selection')
    import plotly.express as px

    correMtr = filtered_data.iloc[:, 2:].corr()  # Exclude 'rec_id' and 'datetime'

    # Create a heatmap using Plotly
    fig = px.imshow(correMtr,
                    labels=dict(x="Attributes", y="Attributes", color="Correlation"),
                    x=correMtr.columns,
                    y=correMtr.columns,
                    color_continuous_scale='Viridis')

    fig.update_layout(title="Correlation matrix of attributes",
                      width=800,
                      height=600)

    # Show the Plotly figure using st.plotly_chart
    st.plotly_chart(fig)

    correMtr2 = model_data.corr()  # Exclude 'rec_id' and 'datetime'

    st.write("We removed the following columns as they are not useful inputs for our model:")
    st.write("- 'rec_id': Record identifier")
    st.write("- 'datetime': Date and time information")
    st.write("- 'season': Removed because it is heavily linked to 'month'")
    st.write("- 'atemp': Removed because it is heavily linked to 'temp'")

    model5_data = filtered_data.drop(['rec_id', 'datetime', 'season', 'atemp', 'casual', 'registered'], axis=1)

    correMtr3 = model5_data.corr() 
    # Create a heatmap using Plotly
    fig = px.imshow(correMtr3,
                    labels=dict(x="Attributes", y="Attributes", color="Correlation"),
                    x=correMtr3.columns,
                    y=correMtr3.columns,
                    color_continuous_scale='Viridis')

    fig.update_layout(title="Correlation matrix of attributes (inputs and outputs)",
                      width=800,
                      height=600)

    # Show the Plotly figure using st.plotly_chart
    st.plotly_chart(fig)

    st.write("- The updated correlation matrix shows that we succesfully checked for multicolinearity.")

st.sidebar.header('Model Prediction steps')

feature_names = ['year', 'month', 'hr', 'holiday', 'weekday', 'workingday', 'weather_condition', 'temp', 'humidity', 'windspeed', 'daylight_hours', 'lag_1', 'lag_24']
def split_data(df, target_columns, test_size=0.3, random_state=42, scale_data=False):
    X = df.drop(columns=target_columns.columns)
    y = target_columns
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    if scale_data:
        # Initialize the StandardScaler
        scaler = StandardScaler()
        
        continuous_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
        X_train[continuous_columns] = scaler.fit_transform(X_train[continuous_columns])
        X_test[continuous_columns] = scaler.transform(X_test[continuous_columns])

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(model_data, model_data[['total_count']], scale_data = False)

def plot_xgb_results(best_xgb_model, feature_names, X_test, y_test, feature_importance_height=400, feature_importance_width=800):
    # Predict using the best XGBoost model
    y_pred = best_xgb_model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    # Feature Importance Plot
    feature_importances = best_xgb_model.feature_importances_
    sorted_feature_importance = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)

    fig_feature_importance = px.bar(
        x=[x[1] for x in sorted_feature_importance],
        y=[x[0] for x in sorted_feature_importance],
        orientation='h',
        labels={'x': 'Feature Importance', 'y': 'Features'},
        title='XGBoost Feature Importance'
    )

    # Set the dimensions of the first graph
    fig_feature_importance.update_layout(
        height=feature_importance_height,
        width=feature_importance_width,
    )

    # Actual vs. Predicted Distribution Plot
    fig_distribution = go.Figure()

    fig_distribution.add_trace(go.Histogram(x=y_test, name='Actual', opacity=0.75, marker_color='blue'))
    fig_distribution.add_trace(go.Histogram(x=y_pred, name='Predicted', opacity=0.75, marker_color='green'))

    fig_distribution.update_layout(
        title='Distribution of Actual vs. Predicted Values',
        xaxis_title='Value',
        yaxis_title='Frequency'
    )

    # Display both plots
    st.plotly_chart(fig_feature_importance)
    st.plotly_chart(fig_distribution)

# Checkboxes for different sections
model_type_determination = st.sidebar.checkbox('Model type determination')
model_optimization = st.sidebar.checkbox('Model optimization')
model_evaluation = st.sidebar.checkbox('Model evaluation')
filtered_data['daylight_hours']=filtered_data.daylight_hours.astype('category')

if model_type_determination: 
    st.subheader('Model type determination')
    data = {
        'Model': ['Decision Tree Regressor', 'Gradient Boosting Regressor', 'AdaBoost Regressor', 'XGB Regressor'],
        'MAE': [28.352003, 29.550290, 56.166602, 20.501716],
        'MSE': [2112.432134, 1975.644440, 4848.004143, 1029.938600],
        'R2': [0.901076, 0.907481, 0.772970, 0.951768]
    }

    df = pd.DataFrame(data)
    st.table(df)
    st.write('The XGB model has the overall best metrics (R2 and MAE). We move forward with this model.')

if model_optimization:
    st.subheader('Model optimization')
    st.write("These are the hyperparameters for the selected model.")
    # Create a DataFrame with the hyperparameters
    hyperparameters = {
        'Hyperparameter': ['gamma', 'learning_rate', 'max_depth', 'n_estimators'],
        'Value': [0.12, 0.10, 6.00, 250.00]
    }

    # Create a DataFrame
    hyperparameters_df = pd.DataFrame(hyperparameters)
    st.table(hyperparameters_df)

    st.write('We also include early stopping and performance monitoring to prevent overfitting and optimize training time.')

    # Streamlit app

if model_evaluation:
    st.subheader('Model evaluation')

    data = {
    'Metric': ['R-squared (R2)', 'Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)'],
    'Value': [0.955881, 942.114247, 19.436258, 30.693880]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Streamlit app
    st.table(df)    
    # Load the saved XGBoost model from the pickle file
    with open('xgb_model.pkl', 'rb') as model_file:
        loaded_xgb_model = pickle.load(model_file)

    # Load the best hyperparameters from the pickle file
    with open('best_hyperparameters.pkl', 'rb') as params_file:
        loaded_best_params = pickle.load(params_file)

    # Call the plot_xgb_results function
    plot_xgb_results(loaded_xgb_model, feature_names, X_test, y_test)

def preprocess_inputs(year, month, hr, holiday, weekday, workingday, weather_condition, temp, humidity, windspeed, lag_1, lag_24):
    # Map year to 0 and 1
    year_mapping = {2011: 0, 2012: 1}
    month = max(1, min(12, month))
    hr = max(0, min(23, hr))
    holiday = 1 if holiday.lower() == 'is' else 0
    weekday_mapping = {
        'monday': 0,
        'tuesday': 1,
        'wednesday': 2,
        'thursday': 3,
        'friday': 4,
        'saturday': 5,
        'sunday': 6
    }
    weekday = weekday_mapping.get(weekday.lower(), -1)  # -1 indicates an invalid input
    # Ensure weekday is in the range [0, 6]
    weekday = max(0, min(6, weekday))
    # Map workingday to 0 and 1
    workingday_mapping = {
        'yes': 1,
        'no': 0
    }
    # Use the mapping to convert the user input to the corresponding value
    workingday = workingday_mapping.get(workingday.lower(), -1)  # -1 indicates an invalid input
    # Ensure workingday is either 0 or 1
    workingday = max(0, min(1, workingday))
    weather_condition_mapping = {
        'clear': 1,
        'mist': 2,
        'light snow': 3,
        'heavy rain': 4
    }
    # Use the mapping to convert the user input to the corresponding value
    weather_condition = weather_condition_mapping.get(weather_condition.lower(), -1)  # -1 indicates an invalid input
    weather_condition = max(1, min(4, weather_condition))
    temp = temp / 100.0
    humidity = humidity / 100.0
    # Normalize wind speed by dividing by 67 (max wind speed)
    windspeed = windspeed / 67.0
    lag_1 = lag_1  # Lagged value 1
    lag_24 = lag_24  # Lagged value 24
    
    # Calculate daylight hours based on the hour input
    if 0 <= hr <= 7:
        daylight_hours = 0  # Night
    elif 8 <= hr <= 17:
        daylight_hours = 1  # Day
    else:
        daylight_hours = 0  # Night

    # Create a DataFrame with user inputs
    user_inputs = pd.DataFrame({
        'year': [year_mapping.get(year, 0)],
        'month': [month],
        'hr': [hr],
        'holiday': [holiday],
        'weekday': [weekday],
        'workingday': [workingday],
        'weather_condition': [weather_condition],
        'temp': [temp],
        'humidity': [humidity],
        'windspeed': [windspeed],
        'lag_1': [lag_1],
        'lag_24': [lag_24],
        'daylight_hours': [daylight_hours]
    })

    # Feature scaling (only for numerical features)
    scaler = StandardScaler()
    user_inputs[['temp', 'humidity', 'windspeed', 'lag_1', 'lag_24', 'daylight_hours']] = scaler.fit_transform(user_inputs[['temp', 'humidity', 'windspeed', 'lag_1', 'lag_24', 'daylight_hours']])
    desired_order = ['year', 'month', 'hr', 'holiday', 'weekday', 'workingday', 'weather_condition', 'temp', 'humidity', 'windspeed', 'daylight_hours', 'lag_1', 'lag_24']
    user_inputs = user_inputs[desired_order]

    return user_inputs

# Function to make predictions using the trained model
def predict_bike_count(trained_model, user_inputs):
    # Make predictions using the trained model
    predicted_count = trained_model.predict(user_inputs)

    return predicted_count

if st.sidebar.checkbox('Bike count prediction'):
    st.subheader('Bike count prediction')
    # User inputs
    user_year = st.slider('Select Year', 2011, 2012)
    user_month = st.slider('Select Month', 1, 12)
    user_hr = st.slider('Select Hour', 0, 23)
    user_holiday = st.radio('Is it a Holiday?', ('Yes', 'No'))
    user_weekday = st.selectbox('Select Weekday', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
    user_workingday = st.radio('Is it a Working Day?', ('Yes', 'No'))
    user_weather_condition = st.selectbox('Select Weather Condition', ('Clear', 'Mist', 'Light Snow', 'Heavy Rain'))
    user_temp_input = st.slider('Enter Temperature (°C)', 0.0, 41.0, 20.0)
    user_humidity_input = st.slider('Enter Humidity (%)', 0.0, 100.0, 50.0)
    user_windspeed_input = st.slider('Enter Wind Speed (mph)', 0.0, 67.0, 20.0)
    user_lag_1 = st.slider('Enter the bike count for the previous hour', 0, 100, 0)
    user_lag_24 = st.slider('Enter the bike count for 24 hours ago', 0, 100, 0)


    user_inputs = preprocess_inputs(user_year, user_month, user_hr, user_holiday, user_weekday, user_workingday, user_weather_condition, user_temp_input, user_humidity_input, user_windspeed_input, user_lag_1, user_lag_24)
    
    with open('xgb_model.pkl', 'rb') as model_file:
        loaded_xgb_model = pickle.load(model_file)

    predicted_count = predict_bike_count(loaded_xgb_model, user_inputs)

    st.success(f'Predicted bike count for the selected parameters: {predicted_count[0]:,.0f} bikes')

st.sidebar.header('Business case')
if st.sidebar.checkbox('Business application'):
    st.subheader('Business application')
    st.write("Our bike prediction model leverages data from two years and offers interval-based predictions for the number of bikes based on various features and data points. Here are the key features of our model:")

    # Bullet Points
    st.write("- Combines data from two years, with the second year showing consistently higher average bike counts.")
    st.write("- Accounts for unpredictable events like strikes that can turn regular working days into holidays.")
    st.write("- Incorporates a wide range of features, including historical bike counts, data from the previous hour, and data from the same time on the previous day.")
    st.write("- Provides an interval range for the number of bikes instead of a single point estimate to enhance prediction accuracy.")
    st.write("- This interval range acknowledges the inherent uncertainty in bike demand, ensuring forecasts are robust and adaptable to potential variations and fluctuations.")
    st.write("- Users can make more informed decisions about bike availability and allocation using these interval-based predictions.")
    st.write("- Next steps could be to do models tailored to registered and casual users to pattern unique customer segment behaviors.")

# Footer
st.write('---')
st.write('Created by Team 3')
