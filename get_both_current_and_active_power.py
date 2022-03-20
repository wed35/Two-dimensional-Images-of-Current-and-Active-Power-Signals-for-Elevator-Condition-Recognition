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
   split_colon = str_time.split(':') # [hh, mm, ss.ms]
   split_dot = split_colon[2].split('.') # [ss, ms]
   time_list = []
   for i in range(len(split_colon)-1):
      time_list.append(split_colon[i])
   time_list.append(split_dot[0])
   if len(split_dot)>1:
      time_list.append(str(int(int(split_dot[1])/100000)))
   else:
      time_list.append('0')
   return time_list

def check_second(time_list, mode):
  temp0, temp1, temp2 = int(time_list[0]), int(time_list[1]), int(time_list[2])
  ms = time_list[3]

  if mode=='-1':
    temp2 = int(time_list[2])-1
    if temp2<0:
      temp1 = int(time_list[1]) - 1
      if temp1<0:
        temp0 -= 1
        temp1 += 59
      temp2 += 59

  if mode=='+1':
    temp2 = int(time_list[2])+1
    if temp2>59:
      temp1 += 1
      if temp1>59:
        temp0 += 1
        temp1 -= 59
      temp2 -=59

  time_list = [str(temp0), str(temp1), str(temp2), ms]
  return time_list

def bar_time(time_set): # make time format as hh_mm_ss_ms
   split_colon = time_set.split(':')
   hh = split_colon[0]
   mm = split_colon[1]
   split_dot = split_colon[2].split('.')
   if len(split_dot)==1:
      ss = split_dot[0]
      ms = str(0)
   else:
      ss = split_dot[0]
      ms = str(int(int(split_dot[1])/100000))
   return hh+'_'+mm+'_'+ss+'_'+ms

def colon_time(time_elements): # make time format as hh:mm:ss.ms
   '''
      PARAMETER => ['hh', 'mm', 'ss', 'ms']
   '''
   if time_elements[3]=='0':
      return time_elements[0]+':'+time_elements[1]+':'+time_elements[2]
   else:
      return time_elements[0]+':'+time_elements[1]+':'+time_elements[2]+'.'+time_elements[3]


#%%
# get current
check_time = time.time()
start = '2019-08-09 12:03:00.000'
end = '2019-08-09 12:03:50.000'

df_cur = Select('HisItemCurr', start, end) # 600 data per minute
plt.plot(df_cur['DataSavedTime'], df_cur['Item005'])
plt.ylim(-10, 100)
plt.show()
#cur_result.to_csv('C:/Users/haeng/Desktop/test'+'.csv')
print('time duration = ', time.time()-check_time)


#%%
# get valid current values
check_time = time.time()
cur_date, cur_values = [], []

i = 1
temp_cnt = []
while i in range(1, len(df_cur['Item005'])):
   if df_cur['Item005'][i-1]==0 and df_cur['Item005'][i]!=0:
      cnt_zero = 0
      temp_date = []
      temp_values = []
      temp_date.append(df_cur['DataSavedTime'][i-1])
      temp_values.append(df_cur['Item005'][i-1])
      j = i
      while j in range(i, len(df_cur['Item005'])):
         if df_cur['Item005'][j]!=0 and j+1<=len(df_cur['Item005']-1):
            if df_cur['Item005'][j+1]==0:
               cnt_zero += 1
            else:
               cnt_zero = 0
         elif df_cur['Item005'][j]==0 and j+1<=len(df_cur['Item005']-1):
            if df_cur['Item005'][j+1]!=0:
               cnt_zero = 0
            else:
               cnt_zero += 1
            if cnt_zero>41:
               temp_cnt.append(cnt_zero)
               cnt_zero = 0
               break
         temp_date.append(df_cur['DataSavedTime'][j])
         temp_values.append(df_cur['Item005'][j])
         j += 1

      temp_date.append(df_cur['DataSavedTime'][j])
      temp_values.append(df_cur['Item005'][j])
      i = j
      cur_date.append(temp_date)
      cur_values.append(temp_values)
   i += 1

for i in range(len(cur_date)):
   del cur_date[i][len(cur_date[i])-40:]
   del cur_values[i][len(cur_values[i])-40:]

print('time duration: ', time.time()-check_time)


#%%
# split current date
start_date, start_time, end_time = [], [], [] # hh:mm:ss.ms
start_time_bar, end_time_bar = [], [] # hh_mm_ss_ms

for i in range(len(cur_date)):
   start_date.append(str(cur_date[i][0]).split()[0])

   start_t = str(cur_date[i][0]).split()[1]
   start_time.append(start_t)
   start_time_bar.append(bar_time(start_t))

   end_t = str(cur_date[i][len(cur_date[i])-1]).split()[1]
   end_time.append(end_t)
   end_time_bar.append(bar_time(end_t))

print(start_date)
print(start_time)
print(start_time_bar)


#%%
# set file name to save csv and png
file_names = []
for i in range(len(cur_date)):
   file_name = start_date[i]+'_'+start_time_bar[i]
   file_names.append(file_name)
print(file_names)


#%%
# save current csv and png
for i in range(len(cur_date)):
   cur_start = start_date[i]+' '+start_time[i][:12]
   cur_end = start_date[i]+' '+end_time[i][:12]
   df_cur_save = Select('HisItemCurr', cur_start, cur_end)
   df_cur_save.to_csv('./elevator_label/'+file_names[i]+'.csv')

   plt.figure()
   plt.plot(df_cur_save['DataSavedTime'], df_cur_save['Item005'])
   plt.ylim(-10, 100)
   plt.savefig('./elevator_label/'+file_names[i]+'.png')
   plt.close()


#%%
# get active power by using time of current
# start_, end_ --> xx:xx:xx.xxx

df_act_dict = {}
for i in range(len(cur_date)):
   # change start second by substracting 1
   start_new = check_second([start_time_bar[i].split('_')[0], start_time_bar[i].split('_')[1],
               start_time_bar[i].split('_')[2], start_time_bar[i].split('_')[3]], '-1')
   s_temp = start_date[i]+' '+colon_time(start_new)

   # change end second by adding 1
   end_new = check_second([end_time_bar[i].split('_')[0], end_time_bar[i].split('_')[1],
               end_time_bar[i].split('_')[2], end_time_bar[i].split('_')[3]], '+1')
   e_temp = start_date[i]+' '+colon_time(end_new)

   check_time = time.time()
   df_act = Select('HisItemAct', s_temp, e_temp) # I don't know why this loop takes a long time in this part
   df_act_dict[i] = df_act
   plt.figure()
   plt.plot(df_act['DataSavedTime'], df_act['Item005'])
   plt.ylim(-10, 100)
   plt.show()

   print('time duration(plot) = ', time.time()-check_time)


#%%
# get real active power time

act_start_time, act_end_time = [], []
act_start_idx, act_end_idx = [], []
for z in range(len(cur_date)):
   #print(df_act_dict[z].shape) # 261, 111
   #df_act_dict[z].to_csv('./elevator_label/active_raw_test'+str(z)+'.csv')
   for i in range(1, df_act_dict[z].shape[0]):
      if df_act_dict[z]['Item005'][i-1]==0 and df_act_dict[z]['Item005'][i]!=0:
         act_start_time.append(str(df_act_dict[z]['DataSavedTime'][i-1]).split()[1])
         act_start_idx.append(i-1)
         break

   for i in range(df_act_dict[z].shape[0]-2, int(df_act_dict[z].shape[0]/2), -1):
      if df_act_dict[z]['Item005'][i]!=0 and df_act_dict[z]['Item005'][i+1]==0:
         act_end_time.append(str(df_act_dict[z]['DataSavedTime'][i+1]).split()[1])
         act_end_idx.append(i+1)
         break

print(act_start_idx)
print(act_start_time)
print(act_end_idx)
print(act_end_time)

#%%
# save active power csv and png
for i in range(len(cur_date)):
   df_act_save = df_act_dict[i][act_start_idx[i]:act_end_idx[i]+1]
   df_act_save.to_csv('./elevator_label/'+file_names[i]+'_active.csv')

   plt.figure()
   plt.plot(df_act_save['DataSavedTime'], df_act_save['Item005'])
   plt.ylim(-10, 100)
   plt.savefig('./elevator_label/'+file_names[i]+'_active.png')
   plt.close()


#%%
