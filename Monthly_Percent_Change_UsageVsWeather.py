import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sheet_names = ["Masked", "Masked",...........]
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('Masked', engine='xlsxwriter')

#Reading data
for z in sheet_names:
      
    weather_stn_Sensitive =  pd.read_excel("Masked", converters= {'Weather_Reading_Dt': pd.to_datetime}, sheet_name = z)
    
    weather_stn_Sensitive.index = weather_stn_Sensitive['Weather_Reading_Dt']
    
    winter = weather_stn_Sensitive[weather_stn_Sensitive['maxTemperature']<=68]
    summer =weather_stn_Sensitive[weather_stn_Sensitive['maxTemperature']>68]
    
    '''
    sns.scatterplot(winter['maxTemperature'],winter['Consumption_Usage'])
    plt.title('Winter')
    plt.show()
    
    sns.scatterplot(summer['maxTemperature'],summer['Consumption_Usage'])
    plt.title('Summer')
    plt.show()
    '''
    ###################################################################################Linear Regressmmion##################################################################
    ################################################################Winter############################################################
    
    lin_reg = LinearRegression()
    lin_reg.fit(winter[['maxTemperature']], winter['Consumption_Usage'])
    y_pred_oncurrenttemp =  lin_reg.predict(winter[['maxTemperature']])
    y_pred_ontenYrNormal =  lin_reg.predict(winter[['tenYrNormalMaxTemperature']])
    '''
    print("Winter data R2:",lin_reg.score(winter[['maxTemperature']], winter['Consumption_Usage']))
    
    sns.jointplot('maxTemperature','Consumption_Usage', data = winter, kind= 'reg',height = 8 ).annotate(stats.pearsonr)
    plt.show()
    '''
    ##############Getting % Difference of predictions vs actual by month ########
    ################For Winter##########
    winter_predictions_Actual = pd.concat([winter.reset_index(drop=True),pd.DataFrame(y_pred_oncurrenttemp, columns = ['y_pred_oncurrenttemp']).reset_index(drop=True)], axis = 1)
    winter_predictions_Actual = pd.concat([winter_predictions_Actual.reset_index(drop=True),pd.DataFrame(y_pred_ontenYrNormal, columns = ['y_pred_ontenYrNormal']).reset_index(drop=True)], axis = 1)
    
    winter_predictions_Actual.index  =  winter.index
    x = winter_predictions_Actual.resample('M').apply( lambda x: len(x))
    
    winter_predictions_Actual = winter_predictions_Actual.resample('M')['y_pred_oncurrenttemp','y_pred_ontenYrNormal','Consumption_Usage'].sum()
    
    winter_predictions_Actual = pd.concat([x['maxTemperature'],winter_predictions_Actual], axis = 1)
    winter_predictions_Actual.rename(columns={'maxTemperature':'NoOfDaysMaxTemp<=68'}, inplace=True)
    
    winter_predictions_Actual['Percentage_change'] = np.round(((winter_predictions_Actual['y_pred_oncurrenttemp']-winter_predictions_Actual['y_pred_ontenYrNormal'])/winter_predictions_Actual['y_pred_ontenYrNormal']) * 100,1)
    
    ################################################################Summer############################################################
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(summer[['maxTemperature']], summer['Consumption_Usage'])
    
    y_pred_oncurrenttemp =  lin_reg_2.predict(summer[['maxTemperature']])
    y_pred_ontenYrNormal =  lin_reg_2.predict(summer[['tenYrNormalMaxTemperature']])
    '''
    print("Summer data R2:",lin_reg_2.score(summer[['maxTemperature']], summer['Consumption_Usage']))
    
    sns.jointplot('maxTemperature','Consumption_Usage', data = summer, kind= 'reg',height = 8 ).annotate(stats.pearsonr)
    plt.show()
    '''
    ###############Getting % Difference of predictions vs actual by month ###########
    ################For Summer##########
    summer_predictions_Actual = pd.concat([summer.reset_index(drop=True),pd.DataFrame(y_pred_oncurrenttemp, columns = ['y_pred_oncurrenttemp']).reset_index(drop=True)], axis = 1)
    summer_predictions_Actual = pd.concat([summer_predictions_Actual.reset_index(drop=True),pd.DataFrame(y_pred_ontenYrNormal, columns = ['y_pred_ontenYrNormal']).reset_index(drop=True)], axis = 1)
    
    summer_predictions_Actual.index  =  summer.index
    
    x = summer_predictions_Actual.resample('M').apply( lambda x: len(x))
    
    summer_predictions_Actual = summer_predictions_Actual.resample('M')['y_pred_oncurrenttemp','y_pred_ontenYrNormal','Consumption_Usage'].sum()
    
    summer_predictions_Actual = pd.concat([x['maxTemperature'],summer_predictions_Actual], axis = 1)
    summer_predictions_Actual.rename(columns={'maxTemperature':'NoOfDaysMaxTemp>68'}, inplace=True)
    summer_predictions_Actual['Percentage_change'] = np.round(((summer_predictions_Actual['y_pred_oncurrenttemp']-summer_predictions_Actual['y_pred_ontenYrNormal'])/summer_predictions_Actual['y_pred_ontenYrNormal']) * 100,1)
    
    summer_predictions_Actual = summer_predictions_Actual[summer_predictions_Actual['NoOfDaysMaxTemp>68']>=15]
    winter_predictions_Actual = winter_predictions_Actual[winter_predictions_Actual['NoOfDaysMaxTemp<=68']>15]
    
    total_predictions  = pd.concat([summer_predictions_Actual,winter_predictions_Actual],axis = 0,sort=True).sort_index()
    #total_predictions = total_predictions.drop(['NoOfDaysMaxTemp<=68', 'NoOfDaysMaxTemp>68'],axis = 1)
    total_predictions.reset_index(inplace = True)
    total_predictions.rename(columns={'NoOfDaysMaxTemp>68':'DaysConsidering>68','NoOfDaysMaxTemp<=68':'DaysConsidering<=68','Weather_Reading_Dt':'Meter_Readings_Month'}, inplace=True)
    # Write each dataframe to a different worksheet
    total_predictions.to_excel(writer, sheet_name=z)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

################################################Writing Predictions to SAP HANA Table "MonthlyWeatherImpact"#################
AllSectors_Combined = pd.DataFrame()
for z in sheet_names:
          onesectordata =  pd.read_excel("Masked",sheet_name = z, dtype={0: str} )
          onesectordata.iloc[:,0] = z
          AllSectors_Combined = AllSectors_Combined.append(onesectordata)       
AllSectors_Combined.rename(columns={"Unnamed: 0": "Sector&RateClass"}, inplace = True)
AllSectors_Combined.fillna(value= 0 , inplace=True)

AllSectors_Combined['year'] = pd.DatetimeIndex(AllSectors_Combined['METER_READING_DATE']).year
AllSectors_Combined['month'] = pd.DatetimeIndex(AllSectors_Combined['METER_READING_DATE']).month
AllSectors_Combined = AllSectors_Combined.iloc[:,[0,8,9,2,3,4,5,6,7]]


connection = dbapi.connect(address='Masked', port=Masked, user='Masked', password='Masked')
# create cursor
cursor=connection.cursor()  
cursor.execute('TRUNCATE TABLE Masked')        

for i,row in AllSectors_Combined.iterrows():
      
    sql = "INSERT INTO Masked  VALUES ("+"?,"*(len(row)-1)+"?)"
    
    cursor.execute(sql, tuple(row))

    connection.commit()
    
connection.close()

exit()


