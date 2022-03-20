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
# get active power
start_t = time.time()
start = '2019-07-30 16:26:53.000'
end = '2019-07-30 16:33:00.000'

df_act = Select('HisItemAct', start, end)
plt.plot(df_act['DataSavedTime'], df_act['Item005'])
plt.show()
print('time durtaion = ', time.time()-start_t)








#%%
# get valid active power
start_t = time.time()
act_date, act_values = [], []

i = 1
temp_cnt = []
while i in range(1, len(df_act['Item005'])):
   if df_act['Item005'][i-1]==0 and df_act['Item005'][i]!=0:
      temp_date = []
      temp_values = []
      temp_date.append(df_act['DataSavedTime'][i-1])
      temp_values.append(df_act['Item005'][i-1])
      j = i
      while j in range(i, len(df_act['Item005'])):
         if df_act['Item005'][j]!=0 and j+1<=len(df_act['Item005']-1):
            if df_act['Item005'][j+1]==0:
               cnt_zero += 1
            else:
               cnt_zero = 0
         elif df_act['Item005'][j]==0 and j+1<=len(df_act['Item005']-1):
            if df_act['Item005'][j+1]!=0:
               cnt_zero = 0
            else:
               cnt_zero += 1
            if cnt_zero>36:
               temp_cnt.append(cnt_zero)
               cnt_zero = 0
               break
         temp_date.append(df_act['DataSavedTime'][j])
         temp_values.append(df_act['Item005'][j])
         j += 1

      temp_date.append(df_act['DataSavedTime'][j])
      temp_values.append(df_act['Item005'][j])
      i = j
      act_date.append(temp_date)
      act_values.append(temp_values)
   i += 1

print(len(act_date))
#print(len(act_date[0]))

for i in range(len(act_date)):
   del act_date[i][len(act_date[i])-35:]
   del act_values[i][len(act_values[i])-35:]
#print(len(act_date[0]))
print('time duration = ', time.time()-start_t)



#%%
# save active power csv and png
start_t = time.time()
import shutil

# save png and csv

#file_name = 'active_'+start_date[0]+'_'

file_name = []
for i in range(len(act_date)):
   name_temp = start_date[i]+'_'+start_time[i]+'_active'
   file_name.append(name_temp)

path_p = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_data/'
if os.path.isdir(path_p+start_date[0])==False:
   os.mkdir(path_p+start_date[0])
   os.mkdir(path_p+start_date[0]+'/active_power')
elif os.path.isdir(path_p+start_date[0]+'/active_power')==False:
   os.mkdir(path_p+start_date[0]+'/active_power')
path_name = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_classification/elevator_label/'

for i in range(len(act_date)):
   act_date_df = pd.DataFrame(np.array(act_date[i]).reshape(len(act_date[i]),1), columns=['DataSavedTime'])
   act_values_df = pd.DataFrame(np.array(act_values[i]).reshape(len(act_values[i]),1), columns=['Item005'])
   act_result = pd.concat([act_date_df, act_values_df], axis=1)
   act_result.to_csv(path_name+file_name[i]+'.csv')
   
   fig = plt.figure()
   plt.plot(act_date[i], act_values[i])
   plt.ylim(-10, 100)
   fig.savefig(path_name+file_name[i]+'.png')
   plt.close(fig)
print('time duration = ', time.time()-start_t)

#%%
# send active power data from my computer to github
start_t = time.time()
#os.mkdir('C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_classification/'+'elevator_data')
github_path = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_classification/elevator_data/'
if os.path.isdir(github_path+start_date[0])==False:
   os.mkdir(github_path+start_date[0])
elif os.path.isdir(github_path+start_date[0]+'/active_power')==True:
  shutil.rmtree(github_path+start_date[0]+'/active_power')
shutil.copytree(path_name, github_path+start_date[0]+'/active_power')

print('time duration = ', time.time()-start_t)

#%%
# copy active power data into github
set_date = '2019-07-15'
dest_path = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_classification/elevator_data/'+set_date
src_path = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_data/'+set_date
shutil.rmtree(dest_path)
shutil.copytree(src_path, dest_path)


#%%
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
for i in range(len(act_date)): 
   
   act_date_df = pd.DataFrame(np.array(act_date[i]).reshape(len(act_date[i]),1), columns=['DataSavedTime'])
   act_values_df = pd.DataFrame(np.array(act_values[i]).reshape(len(act_values[i]),1), columns=['Item005'])
   act_result = pd.concat([act_date_df, act_values_df], axis=1)
   act_result.to_csv(github_path+file_name[i]+'_active.csv')
   
   fig = plt.figure()
   plt.plot(act_date[i], act_values[i])
   plt.ylim(-10, 100)
   fig.savefig(github_path+file_name[i]+'_active.png')
   plt.close(fig)

print('time duration = ', time.time()-start_t)

#%%
