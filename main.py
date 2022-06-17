# Import libraries
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns

# Importing different libraries for various metrics calculations
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Importing libraries for machine learning algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost
import xgboost as xgb
from xgboost import XGBRegressor

# Importing the scikit-learn's train_test_split function
from sklearn.model_selection import train_test_split




#---------------------------------#
# Expanding the width of the page
st.set_page_config(page_title='"PV Production Forecasting and Predicting "',
    layout='wide')

# Setting the main title
st.title("PV Production Forecasting and Predicting ")
#---------------------------------#



#---------------------------------#
# Creating the model
def model_streamlit(df_model):
    
    df_model  = pd.read_csv("df_test.csv",parse_dates=True)
    X = df_model.iloc[:,1:-1] # Using all column except for the last column as X and the first timestamp/date
    Y = df_model.iloc[:,-1] # Selecting the last column as Y
    Z = df_model.iloc[:,0:1]

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-splitting_ratio)/100)

    

    # retrieving the shape of dataframe
    st.markdown('**1.2. Data splits**')
    st.write('Shape of the dataframe')
    st.info(df_model.shape)
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X: Features')
    st.info(list(X.columns))
    st.write('Y: Target Variable')
    st.info(Y.name)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=222,
        max_features=3,
        max_depth=4,
        bootstrap=True,
        oob_score=True,
        verbose = 1,
        n_jobs=-1)
    rf.fit(X_train, Y_train)

    xg = XGBRegressor(n_estimators=parameter_n_estimators,
        random_state=222,
        max_features=3,
        max_depth=4,
        learning_rate=0.1,
        colsample_bytree=0.7,
        bootstrap=True,
        verbosity = 1,
        oob_score=True,
        n_jobs=-1)
    xg.fit(X_train, Y_train)

    mrl =  LinearRegression(

        n_jobs=-1)
    mrl.fit(X_train, Y_train)




    st.subheader('2. Visualizations')
    
    # illustrating the dataframes statistics
    st.markdown('**2.1. Statistics**')
    st.write(df_model.describe().T)


    # plotting the first 50 values
    st.markdown('**2.2. Plots**')
    st.write('Plotting first 50 values of target-value:')



    st.line_chart(Y[:50],use_container_width=True)
    




				
    # Random Forest
    st.subheader('3. Model Performance RF')
    st.markdown('**3.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Mean Absolute Error:')
    st.info( "{:.3f}".format(mean_absolute_error(Y_test, Y_pred_test)) )
    st.write('Mean Squared Error:')
    st.info( "{:.3f}".format(mean_squared_error(Y_test, Y_pred_test)) )
    st.write('Root Mean Squared Error:')
    st.info( "{:.3f}".format(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test))) )
    st.write('$R^2$-Score:')
    st.info( "{:.3f}".format(r2_score(Y_test, Y_pred_test)) )

    st.markdown('**3.3. RF Prediction**')
    rf_pred = pd.DataFrame({ 'Samples of Y_test':Y_test[:15], 'Predicted RF':abs(Y_pred_test[:15])})
    st.write(rf_pred)

    
    st.markdown('**3.4. Plot with actual and predicted values**')
    st.line_chart(rf_pred)
  

    # XGboost 
    st.subheader('4. Model Performance XGBoost')
    st.markdown('**4.2. Test set**')
    y_pred_test_xg = xg.predict(X_test)
    st.write('Mean Absolute Error:')
    st.info( "{:.3f}".format(mean_absolute_error(Y_test, y_pred_test_xg)) )
    st.write('Mean Squared Error:')
    st.info( "{:.3f}".format(mean_squared_error(Y_test, y_pred_test_xg)) )
    st.write('Root Mean Squared Error:')
    st.info( "{:.3f}".format(np.sqrt(metrics.mean_squared_error(Y_test, y_pred_test_xg))) )
    st.write('$R^2$-Score:')
    st.info( "{:.3f}".format(r2_score(Y_test, y_pred_test_xg)) )
    
    st.markdown('**4.3. XGBoost Prediction**')
    xg_pred = pd.DataFrame({ 'Samples of Y_test':Y_test[:15], 'Predicted XGBoost':abs(y_pred_test_xg[:15])})
    st.write(xg_pred)

    st.markdown('**4.4. Plot with actual and predicted values**')
    st.line_chart(xg_pred)


    st.subheader('5. Model Performance MLR')
    st.markdown('**5.2. Test set**')
    Y_pred_test_mrl = mrl.predict(X_test)
    st.write('Mean Absolute Error:')
    st.info( "{:.3f}".format(mean_absolute_error(Y_test, Y_pred_test_mrl)) )
    st.write('Mean Squared Error:')
    st.info( "{:.3f}".format(mean_squared_error(Y_test, Y_pred_test_mrl)) )
    st.write('Root Mean Squared Error:')
    st.info( "{:.3f}".format(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test_mrl))) )
    st.write('$R^2$-Score:')
    st.info( "{:.3f}".format(r2_score(Y_test, Y_pred_test_mrl)) )

    st.markdown('**5.3. MLR Prediction**')
    mrl_pred = pd.DataFrame({ 'Samples of Y_test':Y_test[:15], 'Predicted MLR':abs(Y_pred_test_mrl[:15])})
    st.write(mrl_pred)

    
    st.markdown('**5.4. Plot with actual and predicted values**')
    st.line_chart(mrl_pred)


    # checking wich hyperparameters that can be manipulated 
    #st.subheader('5. Model Parameters for RF')
    #st.write(rf.get_params())

    #st.subheader('6. Model Parameters for XGboost')
    #st.write(xg.get_params())

    #st.subheader('7. Model Parameters for MRl')
    #st.write(mrl.get_params())





#---------------------------------#
st.write("""
# Web-Application which predicts and forecast 
Here  we present the three algorithms *Random Forest Regressor*, *Xgboost* and *Multi Linear Regression*.
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload CSV file'):
    uploaded_file = st.sidebar.file_uploader("Hover over the question mark to the right for more info", type=["csv"],
    
    accept_multiple_files=False,
    help='''The uploaded CSV file must be a time series data set.     
        Required structure of the CSV file:     
        First column of dataset must be in date/timestamp format;        
        Remaining columns must be the features and the target value, with the target value being in the last column;      
        Dataset needs to be pre-processed and cleaned for NaN;     
        ''')
    

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    splitting_ratio = st.sidebar.slider(' Split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Hyperparameters'):
    parameter_n_estimators = st.sidebar.slider('n_estimators', 0, 500, 200, 50)



#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df_model = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df_model)
    model_streamlit(df_model)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Click here to explore existing dataframe'):



        
        
        df_model  = pd.read_csv("df_test.csv",parse_dates=True)
           

        st.markdown('Data collected by SINTEF from test-cell on ZEB-lab.')
        st.write(df_model.head(5))

        model_streamlit(df_model)






