import sys
import numpy as np
import pandas as pd
import csv
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

def cleandata(data1):
    df1 = pd.DataFrame(data1, columns = columns)
    df1["PM10^2"] = df1["PM10"]*df1["PM10"]
    df1["PM2.5^2"] = df1["PM2.5"]*df1["PM2.5"]
    df1["SO2^2"] = df1["SO2"]*df1["SO2"]
    df1["NO2^2"] = df1["NO2"]*df1["NO2"]
    df1["PM2.5^3"] = df1["PM2.5"]*df1["PM2.5"]*df1["PM2.5"]
    df1 = df1.replace("-",np.nan) 
    #df1 = df1.drop(columns=['NO',"O3","RH","WS","WD"])
    df1 = df1.drop(columns=['NO', "WS", "WD", "RH"])
    ans = df1.to_numpy()
    return ans

def preprocess(data1, data, columns_index, dim_row, pm25index, num_e = 0):
    x1 = np.empty(shape = ( (len(data1)-9+1) , dim_row * 9),dtype = float)
    y1 = np.empty(shape = ((len(data1)-9+1) , 1),dtype = float)
    
    notfill = []
    isnan = []
    for i in range(len(data1)):
        where_are_NaNs = np.isnan(data1[i,])
        aList = where_are_NaNs.tolist()
        num = int(aList.count (True))
        isnan += [num]
    
    for i in range(len(isnan)):
        if isnan[i] > num_e:
            notfill +=[i]
    ##接著把notfill 裡index 的row全換成 nan row          
    a= np.full([1,dim_row], np.nan) ## nan row
    data1 = pd.DataFrame(data1, columns = columns_index)

    for name in data1.columns:
        data1[name] = data1[name].fillna(data[name].mean())
    data1 = data1.to_numpy()
    for i in range(len(data1)):
        if i in notfill:
            data1[i,] = a
        
    ### x1_zero 紀錄不能用的 row index ，之後要用 delete 剔除
    x1_zero = []
    for i in range(len(data1)):
        check = 0
        for k in range(10):
            if (i+k) < (len(data1)):
                if isnan[i+k]> num_e:
                    check = 1
        if check !=0:
            x1_zero += [i]
            continue
        if (i+9) < len(data1):
            y1[i,0] = data1[i+9,pm25index]
            x1[i,:] = data1[i:(i+9),0:dim_row].T.reshape(1,-1)

    x1 = np.delete(x1, x1_zero, 0)
    y1 = np.delete(y1, x1_zero, 0)
            
    return x1,y1

def merge(data1, data2, data3, data4, data, columns_index, dim_row, pm25index, num_e = 0 ):
    x1, y1 = preprocess(data1, data, columns_index ,dim_row, pm25index , num_e = 0)
    x2, y2 = preprocess(data2, data, columns_index ,dim_row, pm25index , num_e = 0)
    x3, y3 = preprocess(data3, data, columns_index ,dim_row, pm25index , num_e = 0)
    x4, y4 = preprocess(data4, data, columns_index ,dim_row, pm25index , num_e = 0)
    x = np.concatenate((x1, x2, x3, x4), axis = 0)
    y = np.concatenate((y1, y2, y3, y4), axis = 0)
    
    delete_list = []
    k = pd.DataFrame(x)
    e = (k[0] == 0).to_list()
    for i in range(len(e)):
        if e[i] == True:
            delete_list +=[i] 
    x = np.delete(x, delete_list, 0)
    y = np.delete(y, delete_list, 0)
    
    return x,y 

def Normalization(x):
    mean = np.mean(x, axis = 0) 
    std = np.std(x, axis = 0)

    np.save('./data/mean.npy',mean)
    np.save('./data/std.npy',std)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0 :
                x[i][j] = (x[i][j]- mean[j]) / std[j]
    return x

if __name__ == "__main__":
	#test_name, out_name = sys.argv[1], sys.argv[2]
	raw_data1 = np.genfromtxt('train_datas_0.csv', delimiter = ',', encoding='utf8')
	raw_data2 = np.genfromtxt('train_datas_1.csv', delimiter = ',', encoding='utf8') 
	columns = ['SO2', 'NO', 'NOx', 'NO2', 'CO', 'O3', 'THC', 'CH4', 'NMHC', 'PM10','PM2.5', 'WS', 'WD', 'AT', 'RH']
	high_value = ['SO2', 'NOx', 'NO2', 'CO',"O3", 'THC', 'CH4', 'NMHC', 'PM10','PM2.5','AT',"PM10^2", "PM2.5^2","SO2^2","NO2^2","PM2.5^3"]

	columns_new = ['SO2','NOx', 'NO2', 'CO', 'O3', 'THC', 'CH4', 'NMHC', 'PM10','PM2.5', 'AT']

	data1 = raw_data1[1:6553] 
	data2 = raw_data1[6555:] 
	data3 = raw_data2[1:2161] 
	data4 = raw_data2[2210:] 
	data = np.concatenate((data1, data2, data3, data4), axis = 0)
	data = pd.DataFrame(data, columns = columns)

	data = cleandata(data)
	data = pd.DataFrame(data, columns = high_value)
	data1 = cleandata(data1)
	data2 = cleandata(data2)
	data3 = cleandata(data3)
	data4 = cleandata(data4)

	x,y_train_set = merge(data1, data2, data3, data4, data, high_value, 16, pm25index = 9, num_e = 0 )
	x_train_set = Normalization(x)

	dim = x.shape[1] + 1 
	w = np.zeros(shape = (dim, 1 ))

	x_train_set = np.concatenate((np.ones((x_train_set.shape[0], 1 )), x_train_set) , axis = 1).astype(float)

	learning_rate = np.array([[170]] * dim)
	adagrad_sum = np.zeros(shape = (dim, 1 ))
	 
	for T in range(30000):
	    loss = np.power(np.sum(np.power(x_train_set.dot(w) - y_train_set, 2 ))/ x_train_set.shape[0],0.5)
	    gradient = (-2) * np.transpose(x_train_set).dot(y_train_set-x_train_set.dot(w))
	    adagrad_sum += gradient ** 2
	    w = w - learning_rate * gradient / (np.sqrt(adagrad_sum))

	np.save('./data/4.80.npy',w)     ## save weight
