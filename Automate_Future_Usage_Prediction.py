
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD


#Quering future week weather data from teradata
import teradata

#Connecting to Teradata Database
udaExec = teradata.UdaExec(appName="Automate_Future_Usage_Prediction_SQL", version="1.0", logConsole=True)

with udaExec.connect(method="odbc", system="TD1", username="**", password="**", authentication="**") as connect:
 
    #Queries to get data from Teradata
    Future_6Day_Weather_Query = "select Weather_Reading_Dt , avg(relativeHumidity) as avgRelativeHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(tenYrNormalMaxTemperature) as tenYrNormalMaxTemperature \
                                 from  WEATHER_READINGS_HOURLY_TABLE A \
                                 INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS tenYrNormalMaxTemperature FROM  WEATHER_READINGS_TABLE where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_Cd IN ('tenYrNormalMaxTemperature')  and CAST( Reading_Dttm AS DATE) > CAST( CURRENT_TIMESTAMP AS DATE)  AND Weather_site_id = 'KOKC') B \
                                 on A.Weather_Reading_Dt = B.Reading_Dttm \
                                 where  weather_reading_dt  > CAST( CURRENT_TIMESTAMP AS DATE) AND STATION_ID = 'KOKC' AND OBSERVE_TYPE = 'FORECASTED'  \
                                 group by Weather_Reading_Dt ORDER BY Weather_Reading_Dt;"
    Previous_3Day_Weather_Query = "select Weather_Reading_Dt , avg(relativeHumidity) as avgRelativeHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(tenYrNormalMaxTemperature) as tenYrNormalMaxTemperature \
                                  from  WEATHER_READINGS_HOURLY_TABLE A \
                                  INNER JOIN (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS tenYrNormalMaxTemperature  FROM    WEATHER_READINGS_TABLE where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_Cd IN ('tenYrNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) > CURRENT_DATE - interval '4' day  AND Weather_site_id = 'KOKC') B \
                                  on A.Weather_Reading_Dt = B.Reading_Dttm \
                                  where  weather_reading_dt  > CURRENT_DATE - interval '4' day  AND STATION_ID = 'KOKC' AND OBSERVE_TYPE = 'OBSERVED'  \
                                  group by Weather_Reading_Dt ORDER BY Weather_Reading_Dt;"
    #Reading Queries to Dataframes
    
    WeatherData_Future = pd.read_sql(Future_6Day_Weather_Query,connect, parse_dates = ['Weather_Reading_Dt'])
    WeatherData_Previous = pd.read_sql(Previous_3Day_Weather_Query,connect, parse_dates = ['Weather_Reading_Dt'])
    
    WeatherData = WeatherData_Previous.append(WeatherData_Future).reset_index(drop=True)
    
WeatherData['day_of_week'] = WeatherData['Weather_Reading_Dt'].dt.day_name()
WeatherData.index = WeatherData['Weather_Reading_Dt']
WeatherData['Weekends'] = [ 1 if  a == 'Saturday' or a == 'Sunday' else 0 for a in WeatherData['day_of_week'] ]
WeatherData = WeatherData.drop(['Weather_Reading_Dt','day_of_week'],axis = 1)

#Creating lag variables
WeatherData['lagtemp'] = WeatherData['maxTemperature'].shift(1)
WeatherData['lagtemp2'] = WeatherData['maxTemperature'].shift(2)
WeatherData['lagtemp3'] = WeatherData['maxTemperature'].shift(3)

WeatherData = WeatherData.dropna()


sheet_names = ['RESIDENTIAL','RES_OK131_5', 'RES_TimeOfUse','RES_VariablePeakPricing',"RES_GuaranteedFlatBill","COMMERCIAL","COM_TimeOfUse", "COM_Standard","COM_VariablePeakPricing","COM_Residential"]
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('**\\Utility Commercial Services\\2019\\Analyze Projects\\Weather Impact\\Future_6Day_Usage_Predictions.xlsx', engine='xlsxwriter')
for z in sheet_names:
    #Reading data
    weather_stn_Sensitive = pd.read_excel("H:\\Utility Commercial Services\\2019\\Analyze Projects\\Weather Impact\\Usage&_weather_SQL_Data_AllWeatherParameters.xlsx", converters= {'METER_READING_DATE': pd.to_datetime}, sheet_name = z)
    weather_stn_Sensitive['day_of_week'] = weather_stn_Sensitive['Weather_Reading_Dt'].dt.day_name()
    weather_stn_Sensitive.index = weather_stn_Sensitive['Weather_Reading_Dt']
    weather_stn_Sensitive['Weekends'] = [ 1 if  a == 'Saturday' or a == 'Sunday' else 0 for a in weather_stn_Sensitive['day_of_week'] ]
    weather_stn_Sensitive = weather_stn_Sensitive.drop(['Weather_Reading_Dt','day_of_week','Unnamed: 0'],axis = 1)
    
    #Creating lag variables
    weather_stn_Sensitive['lagtemp'] = weather_stn_Sensitive['maxTemperature'].shift(1)
    weather_stn_Sensitive['lagtemp2'] = weather_stn_Sensitive['maxTemperature'].shift(2)
    weather_stn_Sensitive['lagtemp3'] = weather_stn_Sensitive['maxTemperature'].shift(3)
    
    weather_stn_Sensitive = weather_stn_Sensitive.dropna()
    
    #Dividing days as winter and summer days
    winter = weather_stn_Sensitive[weather_stn_Sensitive['maxTemperature']<=63]
    summer =weather_stn_Sensitive[weather_stn_Sensitive['maxTemperature']>=73]
    
    #Deciding to use summer model or winter model (If More no of days are having high temperatures the we use summer model otherwise we use winter model)
    if len(WeatherData[WeatherData['maxTemperature']>=73])>3:
        ############################################################ Parameter Tuned XgBoost Model for summer#########################################
        #Splitting data into Target/Predictors
        Predictors = summer.drop(['Consumption_Usage','NoOfCustomers'], axis=1)
        Target =  summer["Consumption_Usage"]
        # Used our best model parameters found by GridSearchCV
        best_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                     colsample_bynode=1, colsample_bytree=0.7, gamma=0,
                     importance_type='gain', learning_rate=0.09, max_delta_step=0,
                     max_depth=7, min_child_weight=7, missing=None, n_estimators=300,
                     n_jobs=-1, nthread=None, objective='reg:linear', random_state=0,
                     reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                     silent=None, subsample=0.8, verbosity=1)
        
        # Fit our model to previous 2 years to Till Date weather/usage data
        best_model.fit(Predictors, Target,  verbose=True)
        # Make predictions on usage, using future weather data
        usage_predictions = best_model.predict(WeatherData)
    else:
        ############################################################Final XgBoost Model for winter#########################################
        #Splitting data into Target/Predictors
        Predictors = winter.drop(['Consumption_Usage','NoOfCustomers'], axis=1)
        Target =  winter["Consumption_Usage"]
        # Use our best model parameters found by GridSearchCV
        best_model = xgb.XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=1,
                     colsample_bynode=1, colsample_bytree=0.9, gamma=0,
                     importance_type='gain', learning_rate=0.09, max_delta_step=0,
                     max_depth=6, min_child_weight=5, missing=None, n_estimators=300,
                     n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
                     reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                     silent=None, subsample=0.8, verbosity=1)
        
        # Fit our model to previous weather/usage data
        best_model.fit(Predictors, Target,  verbose=True)
            
        # Make predictions on usage using future weather data
        usage_predictions = best_model.predict(WeatherData)
    usage_predictions = pd.DataFrame(usage_predictions, index = WeatherData.index, columns = ['Predicted_Usage'])
    concat = pd.concat([usage_predictions, WeatherData], axis = 1)
        
    concat.to_excel(writer, sheet_name= z)
# Close the Pandas Excel writer and output the Excel file.
writer.save()
exit()

# ============================================Plots-EDA=================================
# sns.scatterplot(winter['maxTemperature'],winter['Consumption_Usage'])
# plt.title('Winter')
# plt.show()
# 
# sns.scatterplot(summer['maxTemperature'],summer['Consumption_Usage'])
# plt.title('Summer')
# plt.show()
# sns.pairplot(winter.loc[:,['maxTemperature', 'tenYrNormalMaxTemperature','avgRelativeHumidity', 'avgWindSpeed','lagtemp', 'lagtemp2', 'lagtemp3','Consumption_Usage']])
# plt.show()
#=============================================================================
###################################################################################Linear Regressmmion##################################################################
###############Winter###################
# Split Train/Test Set
# =============================================================================
# 
# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(winter.drop(['Consumption_Usage','NoOfCustomers'], axis=1), winter["Consumption_Usage"],random_state=10, test_size=0.30)
# 
# lin_reg = LinearRegression()
# lin_reg.fit(X_train_2, y_train_2)
# y_pred =  lin_reg.predict(X_test_2)
# 
# print("Winter data R2:",lin_reg.score(X_test_2, y_test_2))
# 
# from sklearn.metrics import mean_squared_error
# RMSE_Linear_Regression = mean_squared_error(y_test_2,y_pred)**(1/2)
# print('RMSE_Linear_Regression_Winter',RMSE_Linear_Regression)
# 
#################Summer##################
# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(summer.drop(['Consumption_Usage','NoOfCustomers'], axis=1), summer["Consumption_Usage"],random_state=10, test_size=0.30)
# 
# lin_reg = LinearRegression()
# lin_reg.fit(X_train_2, y_train_2)
# y_pred =  lin_reg.predict(X_test_2)
# 
# print("summer data R2:",lin_reg.score(X_test_2, y_test_2))
# 
# from sklearn.metrics import mean_squared_error
# RMSE_Linear_Regression = mean_squared_error(y_test_2,y_pred)**(1/2)
# print('RMSE_Linear_Regression_summer',RMSE_Linear_Regression)

# =============================================================================
# ####################################################################Random forest######################################
# =============================================================================
# Base model
#  from sklearn.ensemble import RandomForestRegressor
#  RandomForest = RandomForestRegressor(n_estimators = 200, random_state = 10,n_jobs = -1)
#  RandomForest.fit(X_train_2,y_train_2)
#  X_test = np.array(X_test_2)
#  
#  # Predicting a new result
#  y_pred = RandomForest.predict(X_test)
#  
#  #MSE
#  from sklearn.metrics import mean_squared_error
#  mean_squared_error(y_test_2,y_pred)**(1/2)
# 
# 
#  
#  max_depth =  [6,7,8,9,10,11,12,13,14]
#  max_features = [5,6,7,8,9]
#  min_samples_split = [2,4,5,50]
#  min_samples_leaf = [1,5,10,15]
#  
#  # Create the random grid
#  random_grid = {
# 'max_depth': max_depth,
# 'max_features': max_features,
# 'min_samples_split': min_samples_split,
# 'min_samples_leaf': min_samples_leaf}
# 
#  RandomForest = RandomForestRegressor(n_estimators = 200, random_state = 10,n_jobs = -1)
#  random_search = RandomizedSearchCV(RandomForest, param_distributions = random_grid, n_iter = 50, cv = 3,random_state=10, n_jobs = -1)
#  random_search.fit(X_train_2,y_train_2)
#  RF = random_search.best_estimator_
#  
#  RF.fit(X_train_2,y_train_2)
#  
#  X_test = np.array(X_test_2)
#  
#  # Predicting a new result
#  y_pred = RF.predict(X_test)
#  #MSE
#  from sklearn.metrics import mean_squared_error
#  mean_squared_error(y_test_2,y_pred)**(1/2)
# 
# =============================================================================

####################################################################XGBoost############################################
##############################Winter############################
# Split Train/Test Set
#X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(winter.drop(['Consumption_Usage','NoOfCustomers'], axis=1), winter["Consumption_Usage"],random_state=10, test_size=0.001)


#######################################Hand tuning for every important parameter##########################
# from sklearn import metrics
# # Create empty array to store results
# results = []
# 
# # Enumerate through different max_depth values and store results
# for max_depth in [2,3,4,5,10,12,15]:
#     clf = xgb.XGBRegressor(max_depth=max_depth)
#     clf.fit(X_train_2, y_train_2,verbose=False)
#     results.append(
#         {
#             'max_depth': max_depth,
#             'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),
#             'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))
#         })
#     
# # Display Results
# max_depth_lr = pd.DataFrame(results).set_index('max_depth').sort_index()
# 
# # Plot Max_Depth Learning Curve
# max_depth_lr.plot(title="Max_Depth Learning Curve")
# #Best max_depth numbers is 4
# 
# #Manually search for best colsample_bytree
# results = []
# for colsample_bytree in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
#     clf = xgb.XGBRegressor(max_depth= 4, colsample_bytree=colsample_bytree)
#     clf.fit(X_train_2, y_train_2,verbose=False)
#     results.append(
#         {
#             'colsample_bytree': colsample_bytree,
#             'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),
#             'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))
#         })
# max_features_df = pd.DataFrame(results).set_index('colsample_bytree').sort_index()
# 
# max_features_df.plot(title="max_features Learning Curve")
# #Best colsample_bytree is 0.6
# 
# 
# 
# # subsample is Minimum number of samples required at each leaf node to make a split
# #Manually search for best subsample
# results = []
# for subsample in [0.6,0.7,0.8,0.9,1]:
#     clf = xgb.XGBRegressor(max_depth= 4, colsample_bytree=0.6, subsample = subsample)
#     clf.fit(X_train_2, y_train_2,verbose=False)
#     results.append(
#         {
#             'subsample': subsample,
#             'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),
#             'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))
#         })
# subsample = pd.DataFrame(results).set_index('subsample').sort_index()
# 
# subsample.plot(title="subsample Learning Curve")
# #Best subsample is 0.8
# 
# #Manually search for best learning_rate
# results = []
# for learning_rate in [0.09,0.1,0.11,0.12,0.13]:
#     clf = xgb.XGBRegressor(max_depth= 4, colsample_bytree=0.6, subsample = 0.8, learning_rate= learning_rate)
#     clf.fit(X_train_2, y_train_2, verbose=False)
#     results.append(
#         {
#             'learning_rate': learning_rate,
#             'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),
#             'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))
#         })
#     
# learning_rate_lr = pd.DataFrame(results).set_index('learning_rate').sort_index()
# learning_rate_lr.plot(title="Learning Rate Learning Curve")
# #Best nlearning_rate no is 0.11
# 
#
# 
#####################################Performing GridSearchCV to Fine Tune the Previously hand tuned parameters
# from sklearn.model_selection import GridSearchCV
# model = xgb.XGBRegressor(n_estimators= 300)
# # Define Parameters
# param_grid = {"max_depth": [3,4,5,6,7],
#               "colsample_bytree": [0.5,0.6,0.7,0.8,0.9],
#               "learning_rate": [0.09,0.1,0.11,0.12,0.13],
#               "subsample":[0.5,0.6,0.7,0.8,0.9],
#               "min_child_weight": [5,6,7,8,9]}
# gs_cv = GridSearchCV(model, param_grid=param_grid, cv = 6, verbose=10, n_jobs=-1 ).fit(X_train_2, y_train_2)
#
# 
# # Best hyperparmeter setting
# gs_cv.best_estimator_
# 
# =============================================================================
##Splitting data into Target/Predictors
#Predictors = winter.drop(['Consumption_Usage','NoOfCustomers'], axis=1)
#Target =  winter["Consumption_Usage"]
## Use our best model parameters found by GridSearchCV
#best_model = xgb.XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=1,
#             colsample_bynode=1, colsample_bytree=0.9, gamma=0,
#             importance_type='gain', learning_rate=0.09, max_delta_step=0,
#             max_depth=6, min_child_weight=5, missing=None, n_estimators=300,
#             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#             silent=None, subsample=0.8, verbosity=1)
#
## Fit our model to the training set
#best_model.fit(Predictors, Target,  verbose=True)
#
## Make predictions with test data
#predictions = best_model.predict(X_test_2)

####################################################Plot feature importance chart############################
#fig, ax = plt.subplots(figsize=(10,20))
#xgb.plot_importance(best_model, importance_type='gain',ax=ax, show_values = False, grid  = False )
#plt.show()
#
#RMSE_XGBoost = mean_squared_error(y_test_2,predictions)**(1/2)
#
#predictions_winter = pd.DataFrame(predictions, columns = ['Predicted_Usage'] )

##############################Summer_Model############################
# Split Train/Test Set

#X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(summer.drop(['Consumption_Usage','NoOfCustomers'], axis=1), summer["Consumption_Usage"],random_state=10, test_size=0.30)

# =============================================================================
# # Create empty array to store results
# results = []
#  
#  # Enumerate through different max_depth values and store results
# for max_depth in [2,3,4,5,6,7,8,9,10,12,15]:
#      clf = xgb.XGBRegressor(max_depth=max_depth)
#      clf.fit(X_train_2, y_train_2,verbose=False)
#      results.append(
#          {
#              'max_depth': max_depth,
#              'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),
#              'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))
#          })
#      
# # Display Results
# max_depth_lr = pd.DataFrame(results).set_index('max_depth').sort_index()
#  
# # Plot Max_Depth Learning Curve
# max_depth_lr.plot(title="Max_Depth Learning Curve")
# #Best max_depth numbers is 4
#  
# #Manually search for best colsample_bytree
# results = []
# for colsample_bytree in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
#  clf = xgb.XGBRegressor(max_depth= 4, colsample_bytree=colsample_bytree)
#  clf.fit(X_train_2, y_train_2,verbose=False)
#  results.append(
#      {
#          'colsample_bytree': colsample_bytree,
#          'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),
#          'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))
#      })
# max_features_df = pd.DataFrame(results).set_index('colsample_bytree').sort_index()
#  
# max_features_df.plot(title="max_features Learning Curve")
# #Best colsample_bytree is 0.6
#  
#  
#  
# # subsample is Minimum number of samples required at each leaf node
# #Manually search for best subsample
# results = []
# for subsample in [0.6,0.7,0.8,0.9,1]:
#      clf = xgb.XGBRegressor(max_depth= 4, colsample_bytree=0.6, subsample = subsample)
#      clf.fit(X_train_2, y_train_2,verbose=False)
#      results.append(
#          {
#              'subsample': subsample,
#              'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),
#              'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))
#          })
# subsample = pd.DataFrame(results).set_index('subsample').sort_index()
#  
# subsample.plot(title="subsample Learning Curve")
# #Best subsample is 0.8
#  
# #Manually search for best learning_rate
# results = []
# for learning_rate in [0.09,0.1,0.11,0.12,0.13]:
#      clf = xgb.XGBRegressor(max_depth= 4, colsample_bytree=0.6, subsample = 0.8, learning_rate= learning_rate)
#      clf.fit(X_train_2, y_train_2, verbose=False)
#      results.append(
#          {
#              'learning_rate': learning_rate,
#              'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),
#              'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))
#          })
#      
# learning_rate_lr = pd.DataFrame(results).set_index('learning_rate').sort_index()
# learning_rate_lr.plot(title="Learning Rate Learning Curve")
# #Best nlearning_rate no is 0.11
#  
#  
# #For Best N_Estimators Parameter
#  
# results = []
#  
# for n_estimators in [50,60,70,80,90,100,110,150,200,300,400]:
#      clf = xgb.XGBRegressor(max_depth= 4, colsample_bytree=0.6, subsample = 0.8, learning_rate= 0.11, n_estimators= n_estimators)
#      clf.fit(X_train_2, y_train_2,  verbose=False)
#      results.append(
#          {
#              'n_estimators': n_estimators,
#              'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),
#              'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))
#          })
#      
# n_estimators_lr = pd.DataFrame(results).set_index('n_estimators').sort_index()
# n_estimators_lr.plot(title="Estimators Learning Curve")
# #Best Estimator numbers is 200
#  
#  
# #Performing GridSearchCV
# from sklearn.model_selection import GridSearchCV
# model = xgb.XGBRegressor(n_estimators= 300)
# # Define Parameters
# param_grid = {"max_depth": [3,4,5,6,7],
#                "colsample_bytree": [0.5,0.6,0.7,0.8,0.9],
#                "learning_rate": [0.09,0.1,0.11,0.12,0.13],
#                "subsample":[0.5,0.6,0.7,0.8,0.9],
#                "min_child_weight": [5,6,7,8,9]}
# gs_cv = GridSearchCV(model, param_grid=param_grid, cv = 6, verbose=10, n_jobs=-1 ).fit(X_train_2, y_train_2)
#  
#  
#  # Best hyperparmeter setting
# gs_cv.best_estimator_
# 
# =============================================================================
##Splitting data into Target/Predictors
#Predictors = summer.drop(['Consumption_Usage','NoOfCustomers'], axis=1)
#Target =  summer["Consumption_Usage"]
## Use our best model parameters found by GridSearchCV
#best_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#             colsample_bynode=1, colsample_bytree=0.7, gamma=0,
#             importance_type='gain', learning_rate=0.09, max_delta_step=0,
#             max_depth=7, min_child_weight=7, missing=None, n_estimators=300,
#             n_jobs=-1, nthread=None, objective='reg:linear', random_state=0,
#             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#             silent=None, subsample=0.8, verbosity=1)
#
## Fit our model to the training set
#best_model.fit(Predictors, Target,  verbose=True)
#
## Make predictions with test data
#predictions = best_model.predict(X_test_2)

## Plot basic feature importance chart
#fig, ax = plt.subplots(figsize=(10,20))
#xgb.plot_importance(best_model, importance_type='gain',ax=ax, show_values = False, grid  = False )
#plt.show()
#
#RMSE_XGBoost = mean_squared_error(y_test_2,predictions)**(1/2)
#print('RMSE_XGBoost',RMSE_XGBoost)
#
#predictions_summer = pd.DataFrame(predictions, columns = ['Predicted_Usage'] )

# =============================================================================
# ##################################################################################Neural Network####################################################################

# ################################################For Winter####################
# # Split Train/Test Set
# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(winter.drop(['Consumption_Usage'], axis=1), winter["Consumption_Usage"],random_state=10, test_size=0.30)
# 
# winter_Predictors = (X_train_2).values
# winter_target = (y_train_2).values.reshape(-1, 1)
# 
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# winter_Predictors = sc_X.fit_transform(winter_Predictors)
# winter_Predictors_test = sc_X.fit_transform((X_test_2).values)
# 
# early_stopping_monitor = EarlyStopping(patience=2)
# 
# n_cols = winter_Predictors.shape[1]
# 
# model = Sequential()
# model.add(Dense(50,activation='relu',input_shape = (n_cols,)))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
# 
# model.add(Dense(1))
# model.compile(optimizer='adam',loss='mean_squared_error')
# model.fit(winter_Predictors,winter_target,validation_split = 0.3, epochs = 50, callbacks = [early_stopping_monitor])
# 
# predictions = model.predict(winter_Predictors_test)

# predictions_winter = pd.DataFrame(predictions, columns = ['Predicted_Usage'] )
# 
# from sklearn.metrics import mean_squared_error
# RMSE_Neural_network = mean_squared_error(y_test_2,predictions_winter['Predicted_Usage'])**(1/2)
# print('RMSE_Neural_network',RMSE_Neural_network)
# 
# r2_score(y_test_2,predictions)
# 
# ################################################For Summer####################
# summer_Predictors = (summer.loc[:,['maxTemperature', 'avgRelativeHumidity','avgWindSpeed', 'Weekends','lagtemp', 'lagtemp2','lagtemp3']]).values
# summer_target = (summer.loc[:,['Consumption_Usage']]).values
# 
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# summer_Predictors = sc_X.fit_transform(summer_Predictors)
# summer_target = sc_y.fit_transform(summer_target)
# 
# early_stopping_monitor = EarlyStopping(patience=2)
# 
# n_cols = summer_Predictors.shape[1]
# 
# model = Sequential()
# model.add(Dense(100,activation='relu',input_shape = (n_cols,)))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam',loss='mean_squared_error')
# model.fit(summer_Predictors,summer_target,validation_split = 0.3, epochs = 50, callbacks = [early_stopping_monitor])
# 
# predictions_scaled = model.predict(summer_Predictors)
# predictions = sc_y.inverse_transform(predictions_scaled)
# predictions_summer = pd.DataFrame(predictions, columns = ['Predicted_Usage'] )
# 
# from sklearn.metrics import mean_squared_error
# RMSE_Neural_network = mean_squared_error(summer['Consumption_Usage'],predictions_summer['Predicted_Usage'])**(1/2)
# print('RMSE_Neural_network',RMSE_Neural_network)
# 
# def get_new_model():
#     model = Sequential()
#     model.add(Dense(50,activation='relu',input_shape = (n_cols,)))
#     model.add(Dense(50,activation='relu'))
#     model.add(Dense(50,activation='relu'))
#     model.add(Dense(1))
#     return(model)
# # Create list of learning rates: lr_to_test
# lr_to_test = [.000001, 0.01, 1]
# # Loop over learning rates
# for lr in lr_to_test:
#     print('\n\nTesting model with learning rate: %f\n'%lr )
#     
#     # Build new model to test, unaffected by previous models
#     model = get_new_model()
#     
#     # Create SGD optimizer with specified learning rate: my_optimizer
#     my_optimizer = SGD(lr=lr)
#     
#     # Compile the model
#     model.compile(optimizer=my_optimizer,loss='mean_squared_error')
#     
#     # Fit the model
#     model.fit(winter_Predictors,winter_target,validation_split = 0.3, epochs = 50, callbacks = [early_stopping_monitor])
# 
# 
# =============================================================================

##############################linear regression assumptions checking################################
# =============================================================================
# 
# y  = summer.Consumption_Usage
# x  = sm.add_constant(summer.drop(['Consumption_Usage', 'tenYrNormalMaxTemperature','NoOfCustomers'],axis = 1))
# 
# lm = sm.OLS(y,x).fit()
# lm.summary()
# y_Pred =  lm.fittedvalues
# residuals = lm.resid
# ###########################################Checking for assumptions#######################################
# =============================================================================
# =============================================================================
# ####VIF for checking multicollinearity######
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# X_vif = x 
# VIF_Scores = pd.DataFrame([ variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],index = X_vif.columns)
# print("VIF_Scores",VIF_Scores)

####Residuals homoscadasticity checking by plotting Fitted values on x-axis and Residuals on y-axis####
# plt.subplot(1,2,2)
# sns.residplot(y_Pred,y)
# plt.xlabel('Fitted values')
# plt.ylabel('Residuals')

##################Checking  the distribution of residuals(Normality check)######################
# plt.subplot(1,2,1)
# sns.distplot(y-y_Pred,bins=10)
# 
# from statsmodels.graphics.gofplots import ProbPlot
# 
# probplot = sm.ProbPlot(residuals)
# fig = probplot.qqplot()
# 
# fig.set_figheight(5)
# fig.set_figwidth(10)
# 
# fig.axes[0].set_title('Normal Q-Q')
# fig.axes[0].set_xlabel('Theoretical Quantiles')
# fig.axes[0].set_ylabel('Standardized Residuals')
# 
# =============================================================================
