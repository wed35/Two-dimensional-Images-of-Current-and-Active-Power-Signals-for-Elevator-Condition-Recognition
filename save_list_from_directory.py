#%%
# import libraries
import os
import pandas as pd


#%%
labels = ['010', '012', '020', '022', '030', '032', '040', '042', '050', '052', '060', '062',
          '110', '112', '120', '122', '130', '132', '140', '142', '150', '152', '160', '162']
label_num = len(labels)

path = 'C:/Users/haeng/Desktop/HYANG/UNIST/CUop/practice_code/elevator_classification/elevator_label/'
path_label = [path+label+'/' for label in labels]
csv_list = []

# open directories
for i in range(label_numh):
  flist = os.listdir(path_label[i])
  flist_csv = [f for f in flist if f.endswith('.csv')]
  csv_list.append(flist_csv)


#%%
# get current csv and active power csv
cur_file_list, act_file_list = [], []

for i in range(label_num):
  temp_cur_file, temp_act_file = [], []
  for j in range(len(csv_list[i])):
    if '_active.csv' in csv_list[i][j]:
      temp_act_file.append(csv_list[i][j])
    else:
      temp_cur_file.append(csv_list[i][j])
  cur_file_list.append(temp_cur_file)
  act_file_list.append(temp_act_file)

print(len(cur_file_list[0]))
print(len(act_file_list[0]))

#%%
# open current csv
cur_data = {}

for i in range(label_num):
  temp_date, temp_start, temp_end = [], [], []
  cur_data[labels[i]] = []
  for j in range(len(cur_file_list[i])):
    df_cur = pd.read_csv(path_label[i]+cur_file_list[i][j], index_col=0)
    df_cur.columns = ['DataSavedTime', 'Item005']
    temp = df_cur['DataSavedTime']
    temp_date.append(temp[0].split(' ')[0]) # save date
    temp_start.append(temp[0].split(' ')[1]) # save start time
    temp_end.append(temp[df_cur.shape[0]-1].split(' ')[1]) # save end time
  cur_data[labels[i]].append(temp_date)
  cur_data[labels[i]].append(temp_start)
  cur_data[labels[i]].append(temp_end)
  '''
    cur_data[labels[i]][0] --> Date
    cur_data[labels[i]][1] --> Start
    cur_data[labels[i]][2] --> End
  '''

cur_total = 0
for i in range(label_num):
  for j in range(len(cur_data[labels[i]][0])):
    #print(cur_data[labels[i]][0])
    cur_total += 1
  #break
print(cur_total)

#%%
# open active power csv
act_data = {}

for i in range(label_num):
  temp_date, temp_start, temp_end = [], [], []
  act_data[labels[i]] = []
  for j in range(len(act_file_list[i])):
    df_act = pd.read_csv(path_label[i]+act_file_list[i][j], index_col=0)
    df_act.columns = ['DataSavedTime', 'Item005']
    temp = df_act['DataSavedTime']
    temp_date.append(temp[0].split(' ')[0]) # save date
    temp_start.append(temp[0].split(' ')[1]) # save start time
    temp_end.append(temp[df_act.shape[0]-1].split(' ')[1]) # save end time
  act_data[labels[i]].append(temp_date)
  act_data[labels[i]].append(temp_start)
  act_data[labels[i]].append(temp_end)
  '''
    act_data[labels[i]][0] --> Date
    act_data[labels[i]][1] --> Start
    act_data[labels[i]][2] --> End
  '''

act_total = 0
for i in range(label_num):
  for j in act_data[labels[i]][0]:
    act_total += 1
    #print(j)
print(act_total)


#%%
# make dataframe using above dictionaries
before_df = {'Number':[], 'Label':[],
              'Cur_Date':[], 'Cur_Start':[], 'Cur_End':[],
              'Act_Date':[], 'Act_Start':[], 'Act_End':[],
              'Up/Down':[], 'Diff_Floor':[], 'Estimated_Weight':[], 'Labeled_Weight':[]}

before_df['Number'] = [x for x in range(cur_total)]
before_df['Label'] = [labels[x] for x in range(label_num) for y in range(len(cur_file_list[x]))]
before_df['Cur_Date'] = [y for x in labels for y in cur_data[x][0]]
before_df['Cur_Start'] = [y for x in labels for y in cur_data[x][1]]
before_df['Cur_End'] = [y for x in labels for y in cur_data[x][2]]
before_df['Act_Date'] = [y for x in labels for y in act_data[x][0]]
before_df['Act_Start'] = [y for x in labels for y in act_data[x][1]]
before_df['Act_End'] = [y for x in labels for y in act_data[x][2]]
before_df['Up/Down'] = [x[0] for x in labels for y in cur_data[x][0]]
before_df['Diff_Floor'] = [x[1] for x in labels for y in cur_data[x][0]]
before_df['Estimated_Weight'] = [0 for x in range(cur_total)]
before_df['Labeled_Weight'] = ["less than 200kg" if int(x[2])<2 else "over than 200kg"
                                                    for x in labels for y in cur_data[x][0]]

df_csv = pd.DataFrame(before_df)

#%%
# save into csv
df_csv.to_csv(path+'labeled_data.csv', index=0)


#%%
