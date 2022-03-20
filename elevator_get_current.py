#%%
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

%matplotlib inline

#%%
import pymysql

def Select(tableName, start, end) :
   sql_query = """ SELECT DataSavedTime, Item005
   FROM """+tableName+"""
   WHERE DataSavedTime between '"""+start+"""' and '"""+end+"""
   '   ORDER BY DataSavedTime ASC """
   # original : """ WHERE DataSavedTime between '2019-07-05 08:48:00.000' and '2019-07-05 09:47:59.900' ORDER BY DataSavedTime ASC """
   conn = pymysql.connect(host='192.168.100.120', user='root',
    password='1234', database='UYeG_Cloud_New', charset='utf8')
   curs = conn.cursor()
   curs.execute(sql_query)
   rows = curs.fetchall()

   list_for_data = list(rows)
   df_Curr = pd.DataFrame(list_for_data).fillna(0)
   df_Curr.columns=['DataSavedTime', 'Item005']
   return df_Curr

#%%
def split_time(str_time):
   split_colon = str_time.split(':')
   split_dot = split_colon[2].split('.')
   time_list = []
   for i in range(len(split_colon)-1):
      time_list.append(split_colon[i])
   time_list.append(split_dot[0])
   if len(split_dot)>1:
      time_list.append(str(int(int(split_dot[1])/100000)))
   else:
      time_list.append('0')
   return time_list

def make_string(time_list):
   time_string = []
   for i in range(0, len(time_list), 4):
      temp = ''
      for j in range(i, i+4):
         if j!=i+3:
            temp = temp+time_list[j]+'_'
         else:
            temp = temp+time_list[j]
      time_string.append(temp)
   return time_string


#%%
# get current
start_t = time.time()
start = '2019-07-30 16:26:53.000'
end = '2019-07-30 16:33:00.000'

df_curr = Select('HisItemCurr', start, end) # 600 data per minute
plt.plot(df_curr['DataSavedTime'], df_curr['Item005'])
plt.show()
#curr_result.to_csv('C:/Users/haeng/Desktop/test'+'.csv')
print('time duration = ', time.time()-start_t)


#%%
# get valid current values
start_t = time.time()
curr_date, curr_values = [], []

i = 1
temp_cnt = []
while i in range(1, len(df_curr['Item005'])):
   if df_curr['Item005'][i-1]==0 and df_curr['Item005'][i]!=0:
      cnt_zero = 0
      temp_date = []
      temp_values = []
      temp_date.append(df_curr['DataSavedTime'][i-1])
      temp_values.append(df_curr['Item005'][i-1])
      j = i
      while j in range(i, len(df_curr['Item005'])):
         if df_curr['Item005'][j]!=0 and j+1<=len(df_curr['Item005']-1):
            if df_curr['Item005'][j+1]==0:
               cnt_zero += 1
            else:
               cnt_zero = 0
         elif df_curr['Item005'][j]==0 and j+1<=len(df_curr['Item005']-1):
            if df_curr['Item005'][j+1]!=0:
               cnt_zero = 0
            else:
               cnt_zero += 1
            if cnt_zero>41:
               temp_cnt.append(cnt_zero)
               cnt_zero = 0
               break
         temp_date.append(df_curr['DataSavedTime'][j])
         temp_values.append(df_curr['Item005'][j])
         j += 1

      temp_date.append(df_curr['DataSavedTime'][j])
      temp_values.append(df_curr['Item005'][j])
      i = j
      curr_date.append(temp_date)
      curr_values.append(temp_values)
   i += 1

print(len(curr_date))
#print(len(curr_date[0]))

for i in range(len(curr_date)):
   del curr_date[i][len(curr_date[i])-40:]
   del curr_values[i][len(curr_values[i])-40:]
#print(len(curr_date[0]))
print('time duration = ', time.time()-start_t)


#%%
# split current date
start_date, start_time, end_time = [], [], []
start_temp, end_temp = [], []
for i in range(len(curr_date)):
   #start_temp, end_temp = [], []

   start_split= str(curr_date[i][0]).split()
   stime_split = start_split[1].split(':')
   start_date.append(start_split[0])
   start_temp.extend(split_time(start_split[1]))
   start_time_bar = make_string(start_temp)
   start_time.append(start_time_bar)

   end_split = str(curr_date[i][len(curr_date[i])-1]).split()
   etime_split = end_split[1].split(':')
   end_temp.extend(split_time(end_split[1]))
   end_time_bar = make_string(end_temp)
   end_time.append(end_time_bar)

#print('start_date = ', start_date)
print('start_time = ', start_time)
#print('end_time = ', end_time)


#%%
# save current csv and png
start_t = time.time()
import shutil

# save png and csv
#file_name = 'current_'+start_date[0]+'_'
path_p = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_data/'
github_path = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_classification/elevator_label/'
file_name = []
for i in range(len(curr_date)):
   name_temp = start_date[i]+'_'+start_time[i]
   file_name.append(name_temp)

'''
if os.path.isdir(path_p+start_date[0])==False:
   os.mkdir(path_p+start_date[0])
   os.mkdir(path_p+start_date[0]+'/current')
elif os.path.isdir(path_p+start_date[0]+'/current')==False:
   os.mkdir(path_p+start_date[0]+'/current')
path_name = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_data/'+start_date[0]+'/current/'
'''
for i in range(len(curr_date)): 
   
   curr_date_df = pd.DataFrame(np.array(curr_date[i]).reshape(len(curr_date[i]),1), columns=['DataSavedTime'])
   curr_values_df = pd.DataFrame(np.array(curr_values[i]).reshape(len(curr_values[i]),1), columns=['Item005'])
   curr_result = pd.concat([curr_date_df, curr_values_df], axis=1)
   curr_result.to_csv(github_path+file_name[i]+'.csv')
   
   fig = plt.figure()
   plt.plot(curr_date[i], curr_values[i])
   plt.ylim(-10, 100)
   fig.savefig(github_path+file_name[i]+'.png')
   plt.close(fig)

print('time duration = ', time.time()-start_t)


#%%
# send current data from my computer to github
start_t = time.time()


if os.path.isdir(github_path+start_date[0])==False:
   shutil.copytree(path_p+start_date[0])
elif os.path.isdir(github_path+start_date[0]+'/current')==True:
   shutil.rmtree(github_path+start_date[0]+'/current')
shutil.copytree(path_name, github_path+start_date[0]+'/current')

print('time duration = ', time.time()-start_t)



#%%
