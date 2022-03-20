#%%
# import labraries
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

#%%
# preprocess time of active power
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
# read labeled data
label_directories = [f for f in os.listdir('./elevator_label/') if not '.' in f] # labels
label_cur, label_act = {}, {}

for i, l in enumerate(label_directories):
	label_data = [f for f in os.listdir('./elevator_label/'+l+'/') if not 'png' in f] # get only csv files
	act_csv = [f for f in label_data if 'active' in f]
	cur_csv = [f for f in label_data if not 'active' in f and not '.ini' in f]
	print(l, " ==> ", len(act_csv)==len(cur_csv))
	
	label_cur[l] = cur_csv
	label_act[l] = act_csv


#%%
# find max time duration
cur_max_duration, act_max_duration = {}, {}
cur_min_duration, act_min_duration = {}, {} # to check error data
error_data = ''

for i, l in enumerate(label_directories):
	path = './elevator_label/'+l+'/'
	max_cur, max_act = 0, 0
	min_cur, min_act = 500, 500
	temp = ''
	for j in range(len(label_cur[l])):
		df_cur = pd.read_csv(path+label_cur[l][j])
		df_act = pd.read_csv(path+label_act[l][j])
		if df_cur.shape[1]<3:
			df_cur.columns = ['DataSavedTime', 'Item005']
		else:
			df_cur.columns = ['Index', 'DataSavedTime', 'Item005']
		cur_time_duration = df_cur.shape[0]
		act_time_duration = df_act.shape[0]
		if max_cur<cur_time_duration:
			max_cur = cur_time_duration
			temp = path+label_cur[l][j]
		if max_act<act_time_duration:
			max_act = act_time_duration
		if min_cur>=cur_time_duration:
			min_cur = cur_time_duration
			error_data = path+label_cur[l][j]
		if min_act>=act_time_duration:
			min_act = act_time_duration
	cur_max_duration[l] = max_cur
	act_max_duration[l] = max_act
	cur_min_duration[l] = min_cur
	act_min_duration[l] = min_act


print('cur_max_duration', '='*20, '\n', cur_max_duration)
print('cur_min_duration', '='*20, '\n', cur_min_duration)
print('act_max_duration', '='*20, '\n', act_max_duration)
print('act_min_duration', '='*20, '\n', act_min_duration)





#%%
# plot graphs of current in one figure
target_label = ['012', '052']

for i, l in enumerate(label_directories):
	if not l in target_label:
		continue
	path = './elevator_label/'+l+'/'
	plt.figure()
	for j in range(len(label_cur[l])):
		df_cur = pd.read_csv(path+label_cur[l][j])
		num_of_rows = df_cur.shape[0]
		if df_cur.shape[1]<3:
			df_cur.columns = ['DataSavedTime', 'Item005']
		else:
			df_cur.columns = ['Index', 'DataSavedTime', 'Item005']

		if num_of_rows!=cur_max_duration[l]:
			difference = cur_max_duration[l]-num_of_rows
			if df_cur.shape[1]<3:
				df_cur['Index'] = pd.Series([k for k in range(num_of_rows+difference)])
				for k in range(difference):
					df_cur[num_of_rows+k, 'Item005'] = 0
			else:
				for k in range(difference):
					df_cur.loc[num_of_rows+k, 'Index']  = num_of_rows+k
					df_cur[num_of_rows+k, 'Item005'] = 0

		# plot data
		if l[2]=='0': # 0 -> L
			plt.title('Label '+l[0:2]+'L')
		else:
			plt.title('Label '+l[0:2]+'H')
		plt.plot(df_cur['Index'], df_cur['Item005'])
		plt.xlabel('Time Duration (millisecond)')
		plt.ylabel('Current')
		plt.ylim(-10, 100)
	plt.savefig('./elevator_label/'+'current_case_'+l+'.png')
	plt.show()
	#plt.close()


#%%
# plot graphs of active power in one figure
for i, l in enumerate(label_directories):
	if not l in target_label:
		continue
	path = './elevator_label/'+l+'/'
	plt.figure()
	for j in range(len(label_act[l])):
		df_act = pd.read_csv(path+label_act[l][j])
		df_act = preprocess_active_power(df_act)

		num_of_rows = df_act.shape[0]
		difference = act_max_duration[l]-num_of_rows
		df_act['Index'] = pd.Series([k for k in range(num_of_rows+difference)])
		for k in range(difference):
			df_act[num_of_rows+k, 'Item005'] = 0

		# plot data
		if l[2]=='0': # 0 -> L
			plt.title('Label '+l[0:2]+'L')
		else:
			plt.title('Label '+l[0:2]+'H')
		plt.plot(df_act['Index'], df_act['Item005'])
		plt.xlabel('Time Duration (millisecond)')
		plt.ylabel('Active Power')
		plt.ylim(-10, 100)
	plt.savefig('./elevator_label/'+'active_power_case_'+l+'.png')
	plt.show()
	#plt.close()

	




#%%
# plot only target labels
target_label = ['012', '052']

# current
for i, l in enumerate(label_directories):
	if not l in target_label:
		continue
	path = './elevator_label/'+l+'/'
	plt.figure()
	for j in range(int(len(label_cur[l])/2)):
		df_cur = pd.read_csv(path+label_cur[l][j])
		num_of_rows = df_cur.shape[0]
		if df_cur.shape[1]<3:
			df_cur.columns = ['DataSavedTime', 'Item005']
		else:
			df_cur.columns = ['Index', 'DataSavedTime', 'Item005']

		if num_of_rows!=cur_max_duration[l]:
			difference = cur_max_duration[l]-num_of_rows
			if df_cur.shape[1]<3:
				df_cur['Index'] = pd.Series([k for k in range(num_of_rows+difference)])
				for k in range(difference):
					df_cur[num_of_rows+k, 'Item005'] = 0
			else:
				for k in range(difference):
					df_cur.loc[num_of_rows+k, 'Index']  = num_of_rows+k
					df_cur[num_of_rows+k, 'Item005'] = 0

		# plot data
		if l[2]=='0': # 0 -> L
			plt.title('Label '+l[0:2]+'L')
		else:
			plt.title('Label '+l[0:2]+'H')
		plt.plot(df_cur['Index'], df_cur['Item005'])
		plt.xlabel('Time_duration (100ms)')
		plt.ylabel('Current')
		plt.ylim(-10, 100)
	plt.show()
	#plt.savefig('./elevator_label/'+'current_case_'+l+'_1.png')

	plt.figure()
	for j in range(int(len(label_cur[l])/2), len(label_cur[l])):
		df_cur = pd.read_csv(path+label_cur[l][j])
		num_of_rows = df_cur.shape[0]
		if df_cur.shape[1]<3:
			df_cur.columns = ['DataSavedTime', 'Item005']
		else:
			df_cur.columns = ['Index', 'DataSavedTime', 'Item005']

		if num_of_rows!=cur_max_duration[l]:
			difference = cur_max_duration[l]-num_of_rows
			if df_cur.shape[1]<3:
				df_cur['Index'] = pd.Series([k for k in range(num_of_rows+difference)])
				for k in range(difference):
					df_cur[num_of_rows+k, 'Item005'] = 0
			else:
				for k in range(difference):
					df_cur.loc[num_of_rows+k, 'Index']  = num_of_rows+k
					df_cur[num_of_rows+k, 'Item005'] = 0

		# plot data
		if l[2]=='0': # 0 -> L
			plt.title('Label '+l[0:2]+'L')
		else:
			plt.title('Label '+l[0:2]+'H')
		plt.plot(df_cur['Index'], df_cur['Item005'])
		plt.xlabel('Time Duration (100ms)')
		plt.ylabel('Current')
		plt.ylim(-10, 100)
	#plt.savefig('./elevator_label/'+'current_case_'+l+'_2.png')
	plt.show()

# active power
for i, l in enumerate(label_directories):
	if not l in target_label:
		continue
	path = './elevator_label/'+l+'/'
	plt.figure()
	for j in range(int(len(label_act[l])/2)):
		df_act = pd.read_csv(path+label_act[l][j])
		df_act = preprocess_active_power(df_act)

		num_of_rows = df_act.shape[0]
		difference = act_max_duration[l]-num_of_rows
		df_act['Index'] = pd.Series([k for k in range(num_of_rows+difference)])
		for k in range(difference):
			df_act[num_of_rows+k, 'Item005'] = 0

		# plot data
		if l[2]=='0': # 0 -> L
			plt.title('Label '+l[0:2]+'L')
		else:
			plt.title('Label '+l[0:2]+'H')
		plt.plot(df_act['Index'], df_act['Item005'])
		plt.xlabel('Time Duration (millisecond)')
		plt.ylabel('Active Power')
		plt.ylim(-10, 100)
	#plt.savefig('./elevator_label/'+'active_power_case_'+l+'_1.png')
	#plt.show()
	#plt.close()

	plt.figure()
	for j in range(int(len(label_act[l])/2), len(label_act[l])):
		df_act = pd.read_csv(path+label_act[l][j])
		df_act = preprocess_active_power(df_act)

		num_of_rows = df_act.shape[0]
		difference = act_max_duration[l]-num_of_rows
		df_act['Index'] = pd.Series([k for k in range(num_of_rows+difference)])
		for k in range(difference):
			df_act[num_of_rows+k, 'Item005'] = 0

		# plot data
		if l[2]=='0': # 0 -> L
			plt.title('Label '+l[0:2]+'L')
		else:
			plt.title('Label '+l[0:2]+'H')
		plt.plot(df_act['Index'], df_act['Item005'])
		plt.xlabel('Time Duration (millisecond)')
		plt.ylabel('Active Power')
		plt.ylim(-10, 100)
	#plt.savefig('./elevator_label/'+'active_power_case_'+l+'_2.png')





#%%
