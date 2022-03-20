#%%
# import labraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import time
from sklearn.datasets import make_blobs
import scipy
import math

%matplotlib inline


#%%
# define functions

# calculated time slice
def del_t(time_series):
  return time_series.shape[0]


def vibration_dot(ts):
  num_of_rows = ts['Item005'].shape[0]
  max_value = ts['Item005'][int(num_of_rows/3):int(num_of_rows*2/3)].max()
  min_value = ts['Item005'][int(num_of_rows/3):int(num_of_rows*2/3)].min()
  criteria = (max_value+min_value)/2

  dot_count = 0
  for i in range(num_of_rows-1):
    if ts['Item005'][i]<=criteria and criteria<=ts['Item005'][i+1]:
      dot_count += 1
    elif ts['Item005'][i]>=criteria and criteria>=ts['Item005'][i+1]:
      dot_count += 1
  return dot_count

def CV(ts, slice):
  return scipy.stats.variation(ts['Item005'][slice:(-1)*slice])

def peak_diff(time_series):
  half_idx = int(len(time_series)/2)
  max1 = time_series[:half_idx].max()
  max2 = time_series[half_idx:].max()
  return abs(max1-max2)/max1


def skewness(df_act):
  # calculate skewness with the adjusted Fisher-Pearson standardized moment coefficient G1
  '''
    n = a set of data points
    g1 = sample skewness(표본 왜도)
    G1 = the adjusted Fisher-Pearson standardized moment coefficient
    분자 = numerator
    분모 = denominator
  '''
  n = df_act.shape[0]
  mean = df_act['Item005'].mean(axis=0)

  # caculate sample skewness, g1
  denominator, numerator = 0, 0
  for i in range(n):
    numerator += math.pow(df_act['Item005'][i]-mean, 3)
    denominator += math.pow(df_act['Item005'][i]-mean, 2)
  numerator = numerator/n
  denominator = denominator/n
  denominator = math.pow(denominator, 3/2)
  g1 = numerator/denominator
  
  # calculate the adjusted Fisher-Pearson standardized moment coefficient
  G1 = g1*math.sqrt(n*(n-1))/(n-2)

  return G1



def quantile(ts, q, cutoff_both_ends):
    if cutoff_both_ends == None :
        return np.quantile(ts['Item005'], q)
    else :
        return np.quantile(ts['Item005'][cutoff_both_ends:-cutoff_both_ends], q)



#--------------------------------------------------------


def two_d_plot(df, x_label, y_label) : 
    plt.scatter(x = df[x_label], y = df[y_label])
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.figure(figsize=(100,100))
    plt.show()

def three_d_plot(df, x_label, y_label, z_label) : 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x_label], df[y_label], df[z_label], marker='o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.show()

def preprocess_active_power(df_act):
  start_idx, end_idx = 0, 0
  for i in range(0, int(df_act.shape[0]/2)):
    if df_act['Item005'][i]==0 and df_act['Item005'][i+1]!=0:
      start_idx = i
      break

  for i in range(df_act.shape[0]-1, int(df_act.shape[0]/2), -1):
    if df_act['Item005'][i]==0 and df_act['Item005'][i-1]!=0:
      end_idx = i
      break

  df_act['DataSavedTime'][:end_idx-start_idx+1] = df_act['DataSavedTime'][start_idx:end_idx+1]
  df_act['Item005'][:end_idx-start_idx+1] = df_act['Item005'][start_idx:end_idx+1]
  df_result = pd.concat([df_act['DataSavedTime'][:end_idx-start_idx+1], df_act['Item005'][:end_idx-start_idx+1]], axis=1)
  return df_result



#%%
# summarize calculation of 2019-07-12 and 2019-07-15
test_time = time.time()
directory_list = [f for f in os.listdir('./elevator_data/') if not '.' in f] # ['2019-07-12', '2019-07-15']
act_csv_list, cur_csv_list = [], []
act_path_list, cur_path_list = [], []

for i in range(len(directory_list)):
  in_directory_list = [f for f in os.listdir('./elevator_data/'+directory_list[i]+'/')] # ['active_power', 'current']
  act_csv_temp = [f for f in os.listdir('./elevator_data/'+directory_list[i]+'/'+in_directory_list[0]+'/') if '.csv' in f]
  cur_csv_temp = [f for f in os.listdir('./elevator_data/'+directory_list[i]+'/'+in_directory_list[1]+'/') if '.csv' in f]
  act_csv_list.extend(act_csv_temp)
  cur_csv_list.extend(cur_csv_temp)
  act_path_list.extend(['./elevator_data/'+directory_list[i]+'/'+in_directory_list[0]+'/'+f for f in act_csv_temp])
  cur_path_list.extend(['./elevator_data/'+directory_list[i]+'/'+in_directory_list[1]+'/'+f for f in cur_csv_temp])
print(len(act_path_list))

#%%
n = len(act_csv_list) # number of data of 2019-07-12 and 2019-07-15
new_cur, new_act = {}, {}
new_cur = pd.DataFrame(new_cur)
new_act = pd.DataFrame(new_act)
for i in range(n):
  '''
  df_cur = pd.read_csv(cur_path_list[i])
  new_cur = new_cur.append({'Index':i, 'del_t':del_t(df_cur), 'CV':CV(df_cur, 30),
                            'Skewness':skewness(df_cur)}, ignore_index=True)
  '''
  df_act = pd.read_csv(act_path_list[i])
  df_act = preprocess_active_power(df_act)
  new_act = new_act.append({'Index':i, 'del_t':del_t(df_act), 'CV':CV(df_act, 30),
                            'Skewness':skewness(df_act), 'Oscillation':vibration_dot(df_act), 'Quantile':quantile(df_act, 0.1, 30)}, ignore_index=True)
  
  #new_cur.to_csv('./clustering/Current_train_all_2.csv', index=False)
  #new_act.to_csv('./clustering/Active_power_train_all_2.csv', index=False)
  
#print('time duration: ', time.time()-test_time)



#%%
act_train = pd.read_csv('./clustering/Active_power_train_all_2.csv')
cur_train = pd.read_csv('./clustering/Current_train_all_2.csv')

#%%
# plot skewness
fig = plt.gcf()
plt.scatter(cur_train['del_t'], act_train['Skewness'])
plt.ylabel('Skewness of Active Power')
plt.xlabel('del_t of Current')
plt.show()
fig.savefig('./elevator_data/no_label_scatter_skewness.png')


#%%
# plot CV
fig = plt.gcf()
plt.scatter(cur_train['del_t'], act_train['CV'])
plt.ylabel('CV of Active Power')
plt.xlabel('del_t of Current')
plt.show()
fig.savefig('./elevator_data/no_label_scatter_CV30.png')

#%%
# plot oscillation
fig = plt.gcf()
plt.scatter(cur_train['del_t'], act_train['Oscillation'])
plt.ylabel('Oscillation of Active Power')
plt.xlabel('del_t of Current')
plt.show()
fig.savefig('./elevator_data/no_label_scatter_Oscillation.png')

#%%
# plot quantile
fig = plt.gcf()
plt.scatter(cur_train['del_t'], act_train['Quantile'])
plt.ylabel('Quantile_0.1 of Active Power')
plt.xlabel('del_t of Current')
plt.show()
fig.savefig('./elevator_data/no_label_scatter_quantile.png')

#%%
# check error data
print(len(act_path_list)==act_train.shape[0])

error_cnt = 0
for i in range(act_train.shape[0]):
  if act_train['Oscillation'][i]>7.5 and act_train['Oscillation'][i]<14:
    #print(act_path_list[i])
    print('Oscillation = ', act_train['Oscillation'][i])
    error_cnt += 1
print(error_cnt)



############## Labeled Data ###################

#%%
# read labeled data
label_directories = [f for f in os.listdir('./elevator_label/') if not '.' in f] # labels
label_cur, label_act = {}, {}

for i, l in enumerate(label_directories):
	label_data = [f for f in os.listdir('./elevator_label/'+l+'/') if not 'png' in f] # get only csv files
	act_csv = [f for f in label_data if 'active' in f]
	cur_csv = [f for f in label_data if not 'active' in f and not '.ini' in f]
	#print(l, " ==> ", len(act_csv)==len(cur_csv))
	
	label_cur[l] = cur_csv
	label_act[l] = act_csv

num_of_files = 0
for i, l in enumerate(label_directories):
  num_of_files += len(label_cur[l])
print(num_of_files)


#%%
# summarzie labeled data
label_act_result, label_cur_result = {}, {}
label_cur_result = pd.DataFrame(label_cur_result, columns=['Index', 'Up/Down', 'floor_diff', 'del_t', 'Skewness'])
label_act_result = pd.DataFrame(label_act_result,
                        columns=['Index', 'Up/Down', 'floor_diff', 'del_t', 'Skewness', 'Oscillation', 'CV', 'Quantile'])
label_idx_path = []


target_label = ['010', '012', '020', '022', '110', '112', '120', '122']
target_label_cur_result, target_label_act_result = {}, {}
target_label_cur_result = pd.DataFrame(target_label_cur_result, columns=['Index', 'Up/Down', 'floor_diff', 'del_t', 'Skewness'])
target_label_act_result = pd.DataFrame(target_label_act_result,
                        columns=['Index', 'Up/Down', 'floor_diff', 'del_t', 'Skewness', 'Oscillation', 'CV', 'Quantile'])

#%%
error_data = {}

cnt = 0
for i, l in enumerate(label_directories): # label_directories
  path = './elevator_label/'+l+'/'
  for j in range(len(label_act[l])):
    '''
    df_cur = pd.read_csv(path+label_cur[l][j])
    if df_cur.shape[1]>2:
      df_cur.columns = ['Index', 'DataSavedTime', 'Item005']
    else:
      df_cur.columns = ['DataSavedTime', 'Item005']
    label_cur_result.loc[cnt] = [cnt, l[0], l[1], del_t(df_cur), skewness(df_cur)]
    '''
    df_act = pd.read_csv(path+label_act[l][j])
    df_act = preprocess_active_power(df_act)
    label_act_result.loc[cnt] = [cnt, l[0], l[1], del_t(df_act), skewness(df_act),
                                 vibration_dot(df_act), CV(df_act, 30), quantile(df_act, 0.1, 30)]
    
    label_idx_path.append(path+label_cur[l][j])
    cnt += 1
#label_cur_result.to_csv('./elevator_label/Label_current_summary.csv', index=False)
label_act_result.to_csv('./elevator_label/Label_active_power_summary.csv', index=False)




#%%
# summarize target labeled data
cnt = 0
for i, l in enumerate(target_label):
  path = './elevator_label/'+l+'/'
  for j in range(len(label_act[l])):
    df_cur = pd.read_csv(path+label_cur[l][j])
    if df_cur.shape[1]>2:
      df_cur.columns = ['Index', 'DataSavedTime', 'Item005']
    else:
      df_cur.columns = ['DataSavedTime', 'Item005']
    target_label_cur_result.loc[cnt] = [cnt, l[0], l[1], del_t(df_cur), skewness(df_cur)]

    df_act = pd.read_csv(path+label_act[l][j])
    df_act = preprocess_active_power(df_act)
    target_label_act_result.loc[cnt] = [cnt, l[0], l[1], del_t(df_act), skewness(df_act),
                                        vibration_dot(df_act), CV(df_act, 30)]
    '''
    if l[0]=='0' and vibration_dot(df_act)>9:
      error_data[cnt] = l+'/'+label_act[l][j]
    if l[0]=='1' and vibration_dot(df_act)<9:
      error_data[cnt] = l+'/'+label_act[l][j]
    '''
    cnt += 1
#print(error_data)
target_label_cur_result.to_csv('./elevator_label/Label_current_summary_for_little_moving.csv', index=False)
#target_label_act_result.to_csv('./elevator_label/Label_active_power_summary_for_little_moving.csv', index=False)



#%%
# plot vibration_dot for little moving
target_label_cur_result = pd.read_csv('./elevator_label/Label_current_summary_for_little_moving.csv')
target_label_act_result = pd.read_csv('./elevator_label/Label_active_power_summary_for_little_moving.csv')


#%%
fig = plt.gcf()
plt.scatter(target_label_cur_result['del_t'][:45], target_label_act_result['Vibration'][:45], color='green')
plt.scatter(target_label_cur_result['del_t'][57:103], target_label_act_result['Vibration'][57:103], color='yellow')
plt.axhline(y=9, color='r')
plt.ylabel('Number of dots')
plt.xlabel('del_t of current')
plt.show()
#plt.close()
fig.savefig('./elevator_label/Label_active_power_vibration_for_little_moving.png')
'''
num_of_error = 0
for i in range(target_label_act_result.shape[0]):
  if target_label_act_result['floor_diff'][i]==1:
    if target_label_act_result['Up/Down'][i]==0 and target_label_act_result['Vibration'][i]>11:
      num_of_error += 1
    if target_label_act_result['Up/Down'][i]==1 and target_label_act_result['Vibration'][i]<11:
      num_of_error += 1
print(num_of_error)
'''

#%%
# plot vibration_dot for all data
label_act_result = pd.read_csv('./elevator_label/Label_active_power_summary.csv')
label_cur_result = pd.read_csv('./elevator_label/Label_current_summary.csv')

#%%
fig = plt.gcf()
plt.scatter(label_cur_result['del_t'][:88], label_act_result['Oscillation'][:88], color='green')
plt.scatter(label_cur_result['del_t'][88:], label_act_result['Oscillation'][88:], color='yellow')
plt.ylabel('Oscillation of Active Power')
plt.xlabel('del_t of Current')
plt.show()
fig.savefig('./elevator_label/Label_active_power_vibration.png')



#%%
# plot CV for target labels
fig = plt.gcf()
plt.scatter(target_label_cur_result['del_t'][:57], target_label_act_result['CV'][:57], color='green')
plt.scatter(target_label_cur_result['del_t'][57:], target_label_act_result['CV'][57:], color='yellow')
plt.ylabel('CV of Active Power')
plt.xlabel('del_t of Currnet')
plt.show()
fig.savefig('./elevator_label/Label_active_power_CV_for_little moving.png')


#%%
# plot CV for all labels
fig = plt.gcf()
plt.scatter(label_cur_result['del_t'][:88], label_act_result['CV'][:88], color='green')
plt.scatter(label_cur_result['del_t'][88:], label_act_result['CV'][88:], color='yellow')
plt.ylabel('CV of Active Power')
plt.xlabel('del_t of Current')
plt.show()

fig.savefig('./elevator_label/Label_active_power_CV.png')

cnt = 0
for i in range(label_act_result.shape[0]):
  if label_act_result['Up/Down'][i]==0 and label_act_result['CV'][i]>=0.4:
    cnt += 1
  if label_act_result['Up/Down'][i]==1 and label_act_result['CV'][i]<0.4:
    cnt += 1
print('number of errors = ', cnt)





#%%
# plot skewness for target labels
fig = plt.gcf()
plt.scatter(target_label_act_result['del_t'][:57], target_label_act_result['Skewness'][:57], color='green')
plt.scatter(target_label_act_result['del_t'][57:], target_label_act_result['Skewness'][57:], color='yellow')
plt.ylabel('Skewness')
plt.xlabel('del_t of active power')
plt.show()
#fig.savefig('./elevator_label/Label_active_power_skewness_for_little_moving.png')




#%%
# plot skewness for all labels
fig = plt.gcf()
plt.scatter(label_act_result['del_t'][:88], label_act_result['Skewness'][:88], color='green')
plt.scatter(label_cur_result['del_t'][88:], label_act_result['Skewness'][88:], color='yellow')
plt.ylabel('Skewness')
plt.xlabel('del_t of active power')
plt.show()

fig.savefig('./elevator_label/Label_active_power_skewness.png')

cnt = 0
for i in range(label_act_result.shape[0]):
  if label_act_result['Up/Down'][i]==0 and label_act_result['Skewness'][i]>=0:
    cnt += 1
  if label_act_result['Up/Down'][i]==1 and label_act_result['Skewness'][i]<0:
    cnt += 1
print('number of errors = ', cnt)



#%%
# plot quantile for all labels
fig = plt.gcf()
plt.scatter(label_act_result['del_t'][:90], label_act_result['Quantile'][:90], color='green')
plt.scatter(label_cur_result['del_t'][90:], label_act_result['Quantile'][90:], color='yellow')
plt.ylabel('Quantile_0.1')
plt.xlabel('del_t of current')
plt.show()
fig.savefig('./elevator_label/Label_active_power_quantile.png')

error_cnt = 0
for i in range(90):
  if label_act_result['Up/Down'][i]==0 and label_act_result['Quantile'][i]<7:
    error_cnt += 1
    print(label_idx_path[i])
print('-'*20)
for i in range(90, label_act_result.shape[0]):
  if label_act_result['Up/Down'][i]==1 and label_act_result['Quantile'][i]>7:
    error_cnt += 1
    print(label_idx_path[i])
print(error_cnt)



























#%%
floors = 6
model = KMeans(n_clusters=2*floors, random_state=0)
train = model.fit(act_train)
centroids = model.cluster_centers_
labels = model.labels_

# figures
cmap_model = np.array(['red', 'lime', 'green', 'orange', 'blue', 'gray', 'magenta', 'cyan', 'purple', 'pink', 'lightblue', 'yellow'])
plt.figure()
plt.scatter(act_train['del_t'], act_train['Skewness'], c=cmap_model[train.labels_], s=10, edgecolors='none')
#plt.scatter(centroids[:, 0], centroids[:, 1], c=cmap_model, marker='x', s=150, linewidths=5, zorder=10)





#%%
cv_const_accur = {}

# ========= SETTING =========
cv_const = 235 # how to set this value?
n_clusters = 12 # number of labels
centers_2 = np.array([[75,0], [75,220],
                      [125,0], [125,220],
                      [160,0], [160,220],
                      [190,0], [190,220],
                      [230,0], [230,220],
                      [270,0], [270,220]])
# ===========================

X_train_2 = np.array([[a, b] for a, b in zip(df_train_2['del_t'], df_train_2['CV']*cv_const)])
kmeans_2 = KMeans(init=centers_2, n_clusters=n_clusters, random_state=0, verbose=0).fit(X_train_2)
y_kmeans_2 = kmeans_2.predict(X_train_2)
plt.scatter(X_train_2[:, 0], X_train_2[:, 1], c=y_kmeans_2, s=50, cmap='viridis')
plt.ylabel('CV * constant(= '+ str(cv_const) + ')' )
plt.xlabel('del_t')
centers_2 = kmeans_2.cluster_centers_
plt.scatter(centers_2[:, 0], centers_2[:, 1], c='black', s=200, alpha=0.5);
print ("number of data points = ", len(df_train_2))

#%%
# call all labels
path = './elevator_label/'
label_directory = [f for f in os.listdir(path) if not '.' in f] # len(label_directory) = num of labels
label_files = {}

for i, l in enumerate(label_directory):
  files = [f for f in os.listdir(path+l)]
  files_act, files_cur = [], []

  for file in files:
    if '.csv' in file:
      if '_active.csv' in file:
        files_act.append(file)
      else:
        files_cur.append(file)
  label_files[l] = [files_act, files_cur]


print(label_directory)

file_cnt = 0
for i, l in enumerate(label_directory):
  for j in range(len(label_files[l][0])):
    file_cnt += 1
print(file_cnt)

file_cnt = 0
for i, l in enumerate(label_directory):
  for j in range(len(label_files[l][1])):
    file_cnt += 1
print(file_cnt)



#%%
# cell for testing
df = pd.DataFrame(np.arange(16).reshape(4,4,), index=['a','b','c','d'], columns=['f','g','h','i'])
'''
   |f|g|h|i
  a|
  b|
  c|
  d|
'''
print(df.sum(axis=0)) # f, g, h, i
print(df.mean(axis=0))
print(df.std(axis=0))
print(df.var(axis=0))
print('='*10)
print(df.sum(axis=1)) # a, b, c, d
print(df.mean(axis=1))
print(df.std(axis=1))
print(df.var(axis=1))
print('='*10)
print(df.min(axis=0))
print(df.max(axis=0))
print(df.min(axis=1))
print(df.max(axis=1))






#%%
