import numpy as np
import cv2

def rectpuls(*arges):
    if len(arges)<2:
        t = arges[0]
        Tp = 1
    else:
        t = arges[0]
        Tp = arges[1]
        
    y = (abs(t)<Tp/2-np.spacing(1)).astype(float)
    y[abs(t+Tp/2)<np.spacing(1)] = 1
    return y

def db(data,*arges):
    option = 0
    if len(arges)>=1:
        if 'power' == arges[0]:
            option = 1
        
    if option:
        result = 10*np.log10(abs(data))
    else:
        result = 20*np.log10(abs(data))
    return result

def cconv(x1,x2,*arges):
    #循环卷积(网上抄的，稍微封装了一下)
    #x1和x2是输入两数组,arges[0]是需要卷积的位数，若无控制则取两则中长的那个
    if len(arges)>0:
        convNum = arges[0]
    else:
        convNum = max(len(x1),len(x2))
    
    x1,x2 = np.array(x1),np.array(x2)
    x1,x2 = np.concatenate((x1,np.zeros(convNum-len(x1)))),np.concatenate((x2,np.zeros(convNum-len(x2))))

    #序列x1周期延拓，以y轴为对称轴对称变换，截取0~N-1的序列
    temp_x1=[]
    temp_x1.append(x1[0])
    for i in range(1,len(x1)):
        temp_x1.append(x1[convNum-i])
    #针对变化后的x1进行0~N-1的循环移动，获得一个N*N的cycle_matrix矩阵
    x1=temp_x1
    cycle_matrix=[]
    cycle_matrix.append(x1)
    for step in range(1,convNum):
        temp=[]
        for i in range(0,convNum):
            temp.append(0)
        for i in range(0,convNum):
            temp[(i+step)%convNum]=x1[i]
        cycle_matrix.append(temp)

    cycle_matrix=np.array(cycle_matrix)
    x2=np.array(x2)
    result=np.matmul(cycle_matrix,np.transpose(x2))
    return result

def batchMark(listData,listLables):
    if len(listData) != len(listLables):
        raise ValueError("two vectors should have the same number")
    result = mark(listData[-1],listLables[-1])
    for i in range(len(listData)-1):
        result = np.concatenate((result,mark(listData[i],listLables[i])),axis = 1)
    return result

def mark(data,label):
    labels = label*np.ones(data.shape[1]).reshape(1,-1)
    return np.concatenate((data,labels))

def mark2D(DataList):
    #要求输入是列表
    FeatureNum = len(DataList)
    key = True
    # helpData = DataList[0][:,0]
    for i in range(FeatureNum-1):
        key = key&(DataList[i].shape == DataList[i+1].shape)
        # helpData = np.concatenate((helpData,DataList[i+1][:,0]),axis = 1)
    if not key:
        raise ValueError('you shold input same feature shape')
    try:SampleNum = DataList[0].shape[1]
    except:return np.array(DataList[0:]).T.reshape(1,DataList[0].shape[0],-1) 
    result = np.zeros(SampleNum*DataList[0].shape[0]*FeatureNum).astype(complex).reshape(SampleNum,DataList[0].shape[0],FeatureNum)
    for i in range(SampleNum):
        for j in range(FeatureNum):
            result[i,:,j] = DataList[j][:,i]
    return result

def BatchMarked2D(listData,listLables):
    if len(listData) != len(listLables):
        raise ValueError("two vectors should have the same number")
    resultData = mark2D(listData[0])
    resultLable = listLables[0]*np.ones(resultData.shape[0])
    for i in range(len(listData)-1):
        resultData = np.concatenate((resultData,mark2D(listData[i+1])),axis = 0)
        if 1 == len(listData[i+1][0].shape):
            a = 1
        else:
            a = listData[i+1][0].shape[1]
        resultLable = np.concatenate((resultLable,listLables[i+1]*np.ones(a)),axis = 0)
    return resultData,resultLable

def crop(fig,axis = 0):
    th1 = 0.8*fig.max()
    th2 = 0.46*fig.max()
    d = []
    e1 = 0
    e2 = -1
    if axis:
        for i in range(int(fig.shape[axis])):
            if(fig[:,i].max()>th1):
                d.append(i)
        for i in range(d[0],0,-1):
            if(fig[:,i].max()<th2):
                e1 = i
                break
        for i in range(d[-1],fig.shape[axis]):
            if(fig[:,i].max()<th2):
                e2 = i
                break
        return cv2.resize(fig[:,e1:e2],(128,128))
    else:
        for i in range(int(fig.shape[axis])):
            if(fig[i,:].max()>th1):
                d.append(i)
        for i in range(d[0],0,-1):
            if(fig[i,:].max()<th2):
                e1 = i
                break
        for i in range(d[-1],fig.shape[axis]):
            if(fig[i,:].max()<th2):
                e2 = i
                break
        return cv2.resize(fig[e1:e2,:],(128,128))