
import teradata
import pandas as pd

#Queries to get data from Teradata
RES_MASKED_ = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND RATE_CATEGORY_CODE = 'OK131-5' AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'R'  Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
RES_TimeOfUse = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND RATE_CATEGORY_CODE in ('OK13T-5','OK13TB-5') AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'R' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
RES_VariablePeakPricing = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND RATE_CATEGORY_CODE in ('OK13V-5','OK13VB-5') AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'R' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
RES_GuaranteedFlatBill = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND RATE_CATEGORY_CODE in ('OK13G-5') AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'R' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
RESIDENTIAL = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'R'  Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
COMMERCIAL = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'C' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
COM_TimeOfUse = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND RATE_CATEGORY_CODE in ( 'OK06T-3','OK06T-5','OK06TB-5') AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'C' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
COM_Standard = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND RATE_CATEGORY_CODE in ( 'OK06-3','OK06-4','OK06-5') AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'C' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
COM_VariablePeakPricing = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND RATE_CATEGORY_CODE in ('OK06V-3','OK06V-5','OK06VB-5') AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'C' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
COM_Residential = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND RATE_CATEGORY_CODE in ( 'OK131-5','OK13G-5','OK13V-5') AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'C' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
OIL = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'O' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
Industrial = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'I' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"
Public_Authority = "select Reading_dttm ,  avgHumidity , avgWindSpeed,  maxTemperature,  TenYearNormalMaxTemperature, Consumption_Usage_kwh, NoOfCustomers from (select Reading_dttm , avg(relativeHumidity) as avgHumidity , avg(windSpeed) as avgWindSpeed, max(temperature) as maxTemperature, avg(TenYearNormalMaxTemperature) as TenYearNormalMaxTemperature from  TABLE(Masked) A INNER JOIN  (SELECT  CAST( Reading_Dttm AS DATE) AS Reading_Dttm, Weather_Reading_Meas AS TenYearNormalMaxTemperature  FROM  TABLE2(Masked) where Weather_Reading_Interval = 'DAILY' AND  Weather_Reading_Type_code IN ('TenYearNormalMaxTemperature') and CAST( Reading_Dttm AS DATE) between '2017-05-01' and CURRENT_DATE AND Weather_site_code = 'KOKC') B on A.Reading_dttm = B.Reading_Dttm where  A.Reading_dttm between '2017-05-01' and CURRENT_DATE AND A.STATION_ID = 'KOKC' AND A.OBSERVE_TYPE = 'OBSERVED' group by Reading_dttm) Z INNER JOIN (SELECT METER_READ_DATE ,SUM(Consumption_Usage_kwh) as Consumption_Usage_kwh, count(distinct a.CONTRACT_ACCOUNT_NUMBER) AS NoOfCustomers  FROM  TABLE3(Masked) a inner join TABLE4(Masked) b on  a.CONTRACT_ACCOUNT_NUMBER =  b.CONTRACT_ACCOUNT_NUMBER  WHERE METER_READ_DATE between '2017-05-01' and CURRENT_DATE AND SUBSTRING(ACCOUNT_Type_CODE FROM 1 FOR 1) = 'P' Group by a.METER_READ_DATE ) Y on Z.Reading_dttm = Y.METER_READ_DATE order by Reading_dttm;"

dictionary = {"RESIDENTIAL":RESIDENTIAL,"RES_MASKED_":RES_MASKED_,"RES_TimeOfUse":RES_TimeOfUse,"RES_VariablePeakPricing":RES_VariablePeakPricing, "RES_GuaranteedFlatBill":RES_GuaranteedFlatBill,"COMMERCIAL":COMMERCIAL, "COM_TimeOfUse":COM_TimeOfUse,"COM_Standard": COM_Standard,"COM_VariablePeakPricing":COM_VariablePeakPricing, "COM_Residential":COM_Residential, "OIL":OIL,"Industrial":Industrial, "Public_Authority":Public_Authority }
writer = pd.ExcelWriter('H:****\\Usage&_weather_SQL_Data_AllWeatherParameters.xlsx', engine='xlsxwriter')

#Connecting to Teradata Database
udaExec = teradata.UdaExec(appName="Automatic_DataLoad_SQL", version="1.0", logConsole=True)

with udaExec.connect(method="odbc", system="**", username="***", password="***", authentication="**") as connect:
 for key, value in dictionary.items():
     
     #Reading Queries to Dataframes
     z = pd.read_sql(value,connect)
     #Exporting Dataframes to excel
     z.to_excel(writer, sheet_name=key)
     
# Close the Pandas Excel writer and output the Excel file.
writer.save()

exit()
