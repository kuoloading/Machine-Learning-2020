import sys
import numpy as np
import pandas as pd
import csv
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_meanstd( data ):
    data = data.to_numpy()
    mean = np.mean(test, axis = 0) 
    std = np.std(test, axis = 0)
    return mean, std

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


def clean_testdata(test,columns1):
    test = test[1:,]
    test = pd.DataFrame(test, columns = columns1)
    test = test.drop(columns=['NO', "WS", "WD", "RH"])
    test = test.replace(0,np.nan) 
    test["PM10^2"] = test["PM10"]*test["PM10"]
    test["PM2.5^2"] = test["PM2.5"]*test["PM2.5"]
    test["SO2^2"] = test["SO2"]*test["SO2"]
    test["NO2^2"] = test["NO2"]*test["NO2"]
    test["PM2.5^3"] = test["PM2.5"]*test["PM2.5"]*test["PM2.5"]
    return test 

def predict(test, dim_row, w):

    columns = ['SO2', 'NO', 'NOx', 'NO2', 'CO', 'O3', 'THC', 'CH4', 'NMHC', 'PM10','PM2.5', 'WS', 'WD', 'AT', 'RH']

    mean_list = [1.715711, 28.391307, 19.720329, 0.860586, 24.012274, 2.502726, 2.159166, 0.351556, 46.099826, 
             24.583638, 24.945175, 2583.665487, 843.580445, 4.645616, 468.634935, 37422.06352]

    high_value = ['SO2', 'NOx', 'NO2', 'CO',"O3", 'THC', 'CH4', 'NMHC', 'PM10','PM2.5','AT',"PM10^2",
              "PM2.5^2","SO2^2","NO2^2","PM2.5^3"]

    a = pd.DataFrame(mean_list)
    a = a.T
    a.columns = high_value 

    for name in test.columns:
        test[name] = test[name].fillna(a[name].mean())
    
    test = test.to_numpy()
    test_x = np.empty(shape = ( 500 , dim_row * 9), dtype = float)
    for i in range(500):
        #if (i+9) < len(test_raw_data):
            test_x[i,:] = test[9*i: 9 * (i+1),0:dim_row].T.reshape(1,-1)

    for i in range(test_x.shape[0]):        ##Normalization
        for j in range(test_x.shape[1]):
            if not std[j] == 0 :
                test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]

    test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)
    answer = test_x.dot(w)
    
    f = open(out_name,"w")
    w = csv.writer(f)
    title = ['id','value']
    w.writerow(title) 
    for i in range(500):
        content = ['id_'+str(i),answer[i][0]]
        w.writerow(content) 


if __name__ == "__main__":

        test_name = sys.argv[1] 
        out_name = sys.argv[2]
        mean  = np.load('./data/mean.npy')
        std = np.load('./data/std.npy')
        columns = ['SO2', 'NO', 'NOx', 'NO2', 'CO', 'O3', 'THC', 'CH4', 'NMHC', 'PM10','PM2.5', 'WS', 'WD', 'AT', 'RH' ]
        mean_list = [1.715711, 28.391307, 19.720329, 0.860586, 24.012274, 2.502726, 2.159166, 0.351556, 46.099826, 24.583638, 24.945175, 2583.665487, 843.580445, 4.645616, 468.634935, 37422.06352]
        high_value = ['SO2', 'NOx', 'NO2', 'CO',"O3", 'THC', 'CH4', 'NMHC', 'PM10','PM2.5','AT',"PM10^2", "PM2.5^2","SO2^2","NO2^2","PM2.5^3"]
        test_pd = np.genfromtxt(test_name, delimiter=',') 
        test = clean_testdata(test_pd , columns)
        #mean, std = get_meanstd(test)
        w = np.load('./data/4.80.npy')

        predict(test, dim_row = 16, w=w)


