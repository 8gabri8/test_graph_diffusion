import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import copy

d_mat = scipy.io.loadmat('RS_10subj.mat') #NB it's a dictionary
print(type(d_mat))
data = d_mat["X_RS"]
print(d_mat["X_RS"].shape)
data_sogg = data[:,:,0] #only one subject 306(nodes)x1190(times)
print(data_sogg.shape)
temp = copy.deepcopy(d_mat)
temp["RS_1"] = data_sogg
del temp["X_RS"]

scipy.io.savemat('RS_1subj.mat', temp)

print(list(temp))

#STRUCTURAL MATRICES
print("#########################")
s_dict = scipy.io.loadmat('SC_avg56.mat') #NB it's a dictionary
#print(s_dict.keys)
s = s_dict["SC_avg56"]
#print(s.shape)
s1 = np.zeros((360,360))
s1[:180,:180] = s[:180,:180] #all sx hemisphere
#print(s1)
s2 = np.zeros((360,360))
s2[180:,180:] = s[180:,180:]

print(s_dict.keys())

temp = copy.deepcopy(s_dict)
del temp["SC_avg56"]
temp["SC_1"] = s1
print(temp.keys())
scipy.io.savemat('SC_1.mat', temp)

temp = copy.deepcopy(s_dict)
del temp["SC_avg56"]
temp["SC_2"] = s2
print(temp.keys())
scipy.io.savemat('SC_2.mat', temp)

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
axes[0].imshow(s1, interpolation='nearest'); 
axes[1].imshow(s2, interpolation='nearest'); 
plt.show()