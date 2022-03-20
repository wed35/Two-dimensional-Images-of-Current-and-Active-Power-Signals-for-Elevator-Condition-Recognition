#%%
# import library
import os
import pandas as pd
import matplotlib.pyplot as plt


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


def get_time(sep_time):
  hh = sep_time[0]
  mm = sep_time[1]
  ss = sep_time[2]
  ms = sep_time[3]
  start_time = hh+':'+mm+':'+ss+'.'+ms
  return start_time

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

def bar_time(time_set):
  split_colon = time_set.split(':')
  hh = split_colon[0]
  mm = split_colon[1]
  split_dot = split_colon[2].split('.')
  ss = split_dot[0]
  ms = str(int(int(split_dot[1])/100))
  return hh+'_'+mm+'_'+ss+'_'+ms


#%%
path = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_classification/elevator_label/'
labels = ['010', '012', '020', '022', '030', '032', '040', '042', '050', '052', '060', '062',
          '110', '112', '120', '122', '130', '132', '140', '142', '150', '152', '160', '162']
label_length = len(labels)
path_label = [path+label+'/' for label in labels]
print(len(path_label))


#%%
labeled_time, valid_path_label = [], []

for i in range(label_length):
  if os.path.isdir(path_label[i])==True:
    valid_path_label.append(path_label[i])
    file_list = os.listdir(path_label[i])
    file_list_csv = [f for f in file_list if f.endswith('.csv')]
    labeled_time.append(file_list_csv)

  else: # no directory for the label
    continue

print(labeled_time[1])
print(valid_path_label)

#%%
date_list, start_list, end_list = [], [], []
original_start_list, original_end_list = [], []

for i in range(len(valid_path_label)):
  temp_dlist, temp_slist, temp_elist = [], [], []
  temp_sorigin, temp_eorigin = [], []
  for j in range(len(labeled_time[i])):
    df_active = pd.read_csv(valid_path_label[i]+labeled_time[i][j], index_col=0)

    temp = df_active[df_active.columns[0]]
    start_temp = temp[0]
    split_words = start_temp.split(' ')
    date = split_words[0] # get date
    split_colon = split_words[1].split(':')
    split_dot = split_colon[2].split('.')
    origin_start = get_time([split_colon[0], split_colon[1], split_dot[0], split_dot[1]])
    start_temp = check_second([split_colon[0], split_colon[1], split_dot[0], split_dot[1]], '-1')
    start_time = get_time(start_temp) # get start time
      
    end_temp = temp[df_active.shape[0]-1]
    split_words = end_temp.split(' ')
    split_colon = split_words[1].split(':')
    split_dot = split_colon[2].split('.')
    origin_end = get_time([split_colon[0], split_colon[1], split_dot[0], split_dot[1]])
    end_temp = check_second([split_colon[0], split_colon[1], split_dot[0], split_dot[1]], '+1')
    end_time = get_time(end_temp) # get end time
        
    temp_dlist.append(date)
    temp_slist.append(start_time)
    temp_elist.append(end_time)
    temp_sorigin.append(origin_start)
    temp_eorigin.append(origin_end)
  date_list.append(temp_dlist)
  start_list.append(temp_slist)
  end_list.append(temp_elist)
  original_start_list.append(temp_sorigin)
  original_end_list.append(temp_eorigin)
'''
print(date_list)
print(start_list)
print(end_list)
'''

#%%
for i in range(len(valid_path_label)):
  for j in range(len(labeled_time[i])):
    s = date_list[i][j]+' '+start_list[i][j]
    e = date_list[i][j]+' '+end_list[i][j]
    sorigin = date_list[i][j]+' '+original_start_list[i][j]
    eorigin = date_list[i][j]+' '+original_end_list[i][j]

    df_act = Select('HisItemAct', s, e)
    fig = plt.figure()
    plt.plot(df_act['DataSavedTime'], df_act['Item005'])
    plt.ylim(-10, 100)

    bar_s = date_list[i][j]+' '+bar_time(original_start_list[i][j])
    df_act.to_csv(valid_path_label[i]+bar_s+'_active.csv')
    fig.savefig(valid_path_label[i]+bar_s+'_active.png')
    plt.close()
  #break
#%%
# get active power for certain label
target_label = ['140', '142', '150', '152', '160', '162']
target_path = [path+tlabel for tlabel in target_label]


# delete original active power files
for i in range(len(target_label)):
  flist = os.listdir(target_path[i])
  origin_active_csv = [file for file in flist if file.endswith('_active.csv')]
  origin_active_png = [file for file in flist if file.endswith('_active.png')]

  if origin_active_csv:
    for f in origin_active_csv:
      os.remove(target_path[i]+'/'+f)
      #print('delete csv')
  
  if origin_active_png:
    for f in origin_active_png:
      os.remove(target_path[i]+'/'+f)
      #print('delete png')

  #break


#%%
target_file_list = []
for i in range(len(target_label)):
  if os.path.isdir(target_path[i])==True: # to check whether the directory exists
    file_list = os.listdir(target_path[i])
    file_list_csv = [f for f in file_list if f.endswith('.csv')]
    target_file_list.append(file_list_csv)

  else: # no directory for the label
    continue

t_date_list, t_start_list, t_end_list = [], [], []
original_t_start_list = []

for i in range(len(target_label)):
  temp_dlist, temp_slist, temp_elist = [], [], []
  temp_sorigin, temp_eorigin = [], []
  for j in range(len(target_file_list[i])):
    df_active = pd.read_csv(target_path[i]+'/'+target_file_list[i][j], index_col=0)

    temp = df_active[df_active.columns[0]]
    start_temp = temp[0]
    split_words = start_temp.split(' ')
    date = split_words[0] # get date
    split_colon = split_words[1].split(':')
    split_dot = split_colon[2].split('.')
    origin_start = get_time([split_colon[0], split_colon[1], split_dot[0], split_dot[1]])
    start_temp = check_second([split_colon[0], split_colon[1], split_dot[0], split_dot[1]], '-1')
    start_time = get_time(start_temp) # get start time
      
    end_temp = temp[df_active.shape[0]-1]
    split_words = end_temp.split(' ')
    split_colon = split_words[1].split(':')
    split_dot = split_colon[2].split('.')
    end_temp = check_second([split_colon[0], split_colon[1], split_dot[0], split_dot[1]], '+1')
    end_time = get_time(end_temp) # get end time
        
    temp_dlist.append(date)
    temp_slist.append(start_time)
    temp_elist.append(end_time)
    temp_sorigin.append(origin_start)
    temp_eorigin.append(origin_end)
  t_date_list.append(temp_dlist)
  t_start_list.append(temp_slist)
  t_end_list.append(temp_elist)
  original_t_start_list.append(temp_sorigin)
#print(original_t_start_list)


#%%
# save active power for certain label
for i in range(len(target_label)):
  for j in range(len(target_file_list[i])):
    s = t_date_list[i][j]+' '+t_start_list[i][j]
    e = t_date_list[i][j]+' '+t_end_list[i][j]
    bar_s = t_date_list[i][j]+'_'+bar_time(original_t_start_list[i][j])

    df_act = Select('HisItemAct', s, e)
    fig = plt.figure()
    plt.plot(df_act['DataSavedTime'], df_act['Item005'])
    plt.ylim(-10, 100)
    fig.savefig(target_path[i]+'/'+bar_s+'_active.png')
    plt.close()

    df_act.to_csv(target_path[i]+'/'+bar_s+'_active.csv')

  #break



#%%
