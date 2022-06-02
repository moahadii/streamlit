
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
import xgboost
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import datasets

# Importing different libraries for various metrics calculations
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split




#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='"PV Production Forecasting and Predicting "',
    layout='wide')

#---------------------------------#

st.title("PV Production Forecasting and Predicting ")

# df_rf  = pd.read_csv("df_test.csv", index_col = 'Timestamp')
#st.dataframe(df_rf.head())

################################################################################################################################

#---------------------------------#
# Page layout
## Page expands to full width


#---------------------------------#
# Model building
def build_model(df_rf):
    
    X = df_rf.iloc[:,1:-1] # Using all column except for the last column as X and the first timestamp/date
    Y = df_rf.iloc[:,-1] # Selecting the last column as Y
    Z = df_rf.iloc[:,0:1]

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-splitting_ratio)/100)


    st.markdown('**1.2. Data splits**')
    st.write('Shape of the dataframe')
    st.info(df_rf.shape)
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
        random_state=parameter_random_state,
        #max_features=parameter_max_features,
        max_features=5,
        max_depth=parameter_max_depth,
        #criterion=parameter_criterion,
        #min_samples_split=parameter_min_samples_split,
        #min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=True,
        oob_score=True,
        verbose = 1,
        n_jobs=-1)
    rf.fit(X_train, Y_train)

    xg = XGBRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        #_SklObjective = "reg:mean_squared_error",
        #max_features=parameter_max_features,
        max_features=5,
        max_depth=parameter_max_depth,
        learning_rate=parameter_learning_rate,
        colsample_bytree=parameter_colsample_bytree,
        bootstrap=True,
        verbosity = 1,
        oob_score=True,
        n_jobs=-1)
    xg.fit(X_train, Y_train)




    st.subheader('2. Visualizations')
    

    st.markdown('**2.1. Statistics**')
    st.write(df_rf.describe().T)



    st.markdown('**2.2. Plots**')
    st.write('Plotting first 50 values of target-value:')



    st.line_chart(Y[:50],use_container_width=True)
    




				

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
  

    st.subheader('4. Model Performance XGboost')

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
    
    st.markdown('**4.3. XGboost Prediction**')
    xg_pred = pd.DataFrame({ 'Samples of Y_test':Y_test[:15], 'Predicted XGboost':abs(y_pred_test_xg[:15])})
    st.write(xg_pred)

    st.markdown('**4.4. Plot with actual and predicted values**')
    st.line_chart(xg_pred)




    st.subheader('5. Model Parameters for RF')
    st.write(rf.get_params())

    st.subheader('6. Model Parameters for XGboost')
    st.write(xg.get_params())






#---------------------------------#
st.write("""
# Web-Application which predicts and forecast 
Here  we present the four algorithms *Random Forest Regressor*, *Xgboost* and *Multi Linear Regression*.
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
    parameter_learning_rate = st.sidebar.slider('learning_rate', 0.1, 1.0, 0.10,0.10)
    #parameter_gamma = st.sidebar.slider('gamma', 0.1, 5.0, 0.1, 0.1)
    #parameter_C = st.sidebar.slider('c', 50, 250, 150, 50)
    parameter_colsample_bytree = st.sidebar.slider('colsample_bytree', 0.1, 0.9, 0.7, 0.1)
    #parameter_max_features = st.sidebar.select_slider('max_features', options=['auto', 'sqrt', 'log2'])
    parameter_max_depth = st.sidebar.select_slider('max_depth', options=[4, 5])
    #parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    #parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 222, 1)
    #parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    #parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    #parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[True,False])
    #parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[-1,1])



#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df_rf = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df_rf)
    build_model(df_rf)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Click here to explore existing dataframe'):



        
        ### alle df over her er endra
        df_rf  = pd.read_csv("df_test.csv",parse_dates=True)
           

        st.markdown('Data collected from test-cell on ZEB-lab, collected by SINTEF.')
        st.write(df_rf.head(5))

        build_model(df_rf)






