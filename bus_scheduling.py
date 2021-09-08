import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bus_35_path = 'C:/Users/90464./Downloads/data/_35_1__201904292221.csv'
holiday_data_path = 'C:/Users/90464./Downloads/data/date_holiday_1.txt'

bus_35 = pd.read_csv(bus_35_path,header=None,
                         names=['id', 'date', 'time', 'bus_stop_id', 'waiting_time', 'on_road_time'])
holiday_data = pd.read_csv(holiday_data_path,header=None,
                          names=['date','holiday'])

holiday_data.head()
bus_35.head()

mean_bus_stop_1 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([1])]['waiting_time'].mean())
mean_bus_stop_2 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([2])]['waiting_time'].mean())
mean_bus_stop_3 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([3])]['waiting_time'].mean())
mean_bus_stop_4 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([4])]['waiting_time'].mean())
mean_bus_stop_5 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([5])]['waiting_time'].mean())
mean_bus_stop_6 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([6])]['waiting_time'].mean())
mean_bus_stop_7 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([7])]['waiting_time'].mean())
mean_bus_stop_8 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([8])]['waiting_time'].mean())
mean_bus_stop_9 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([9])]['waiting_time'].mean())
mean_bus_stop_10 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([10])]['waiting_time'].mean())
mean_bus_stop_11 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([11])]['waiting_time'].mean())
mean_bus_stop_12 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([12])]['waiting_time'].mean())
mean_bus_stop_13 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([13])]['waiting_time'].mean())
mean_bus_stop_14 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([14])]['waiting_time'].mean())
mean_bus_stop_15 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([15])]['waiting_time'].mean())
mean_bus_stop_16 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([16])]['waiting_time'].mean())
mean_bus_stop_17 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([17])]['waiting_time'].mean())
mean_bus_stop_18 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([18])]['waiting_time'].mean())
mean_bus_stop_19 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([19])]['waiting_time'].mean())
mean_bus_stop_20 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([20])]['waiting_time'].mean())
mean_bus_stop_21 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([21])]['waiting_time'].mean())
mean_bus_stop_22 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([22])]['waiting_time'].mean())
mean_bus_stop_23 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([23])]['waiting_time'].mean())
mean_bus_stop_24 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([24])]['waiting_time'].mean())
mean_bus_stop_25 = np.ceil(bus_35[bus_35['bus_stop_id'].isin([25])]['waiting_time'].mean())
mean_bus_stop_21

bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([0]))].index,['waiting_time']]=10
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([1]))].index,['waiting_time']]=mean_bus_stop_1
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([2]))].index,['waiting_time']]=mean_bus_stop_2
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([3]))].index,['waiting_time']]=mean_bus_stop_3
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([4]))].index,['waiting_time']]=mean_bus_stop_4
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([5]))].index,['waiting_time']]=mean_bus_stop_5
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([6]))].index,['waiting_time']]=mean_bus_stop_6
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([7]))].index,['waiting_time']]=mean_bus_stop_7
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([8]))].index,['waiting_time']]=mean_bus_stop_8
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([9]))].index,['waiting_time']]=mean_bus_stop_9
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([10]))].index,['waiting_time']]=mean_bus_stop_10
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([11]))].index,['waiting_time']]=mean_bus_stop_11
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([12]))].index,['waiting_time']]=mean_bus_stop_12
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([13]))].index,['waiting_time']]=mean_bus_stop_13
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([14]))].index,['waiting_time']]=mean_bus_stop_14
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([15]))].index,['waiting_time']]=mean_bus_stop_15
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([16]))].index,['waiting_time']]=mean_bus_stop_16
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([17]))].index,['waiting_time']]=mean_bus_stop_17
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([18]))].index,['waiting_time']]=mean_bus_stop_18
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([19]))].index,['waiting_time']]=mean_bus_stop_19
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([20]))].index,['waiting_time']]=mean_bus_stop_20
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([21]))].index,['waiting_time']]=mean_bus_stop_21
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([22]))].index,['waiting_time']]=mean_bus_stop_22
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([23]))].index,['waiting_time']]=mean_bus_stop_23
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([24]))].index,['waiting_time']]=mean_bus_stop_24
bus_35.loc[bus_35[(bus_35['waiting_time']<0)&(bus_35['bus_stop_id'].isin([25]))].index,['waiting_time']]=mean_bus_stop_25

bus_35.head()
data_bus_35 = pd.merge(bus_35, holiday_data)
data_bus_35.head()

dummy_fields = ['time', 'bus_stop_id', 'holiday']
for each in dummy_fields:
    dummies = pd.get_dummies(data_bus_35[each], prefix=each, drop_first=False)
    data_bus_35 = pd.concat([data_bus_35, dummies], axis=1)

fields_to_drop = ['id', 'date', 'time', 'bus_stop_id', 'holiday']
data = data_bus_35.drop(fields_to_drop, axis=1)
data.head()
quant_features = ['waiting_time', 'on_road_time']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
data.head()
##分离出最后1500条为测试集
test_data = data[-1500:] 
data = data[:-1500]

#将特征和目标变量分开
target_fields = ['waiting_time']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
#取出4500条作为验证集
train_features, train_targets = features[:-3000], targets[:-3000]
val_features, val_targets = features[-3000:], targets[-3000:]
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1/(1+np.exp(-x))  
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 1/（1+np.exp(-x)）  
        #self.activation_function = sigmoid
                    
    
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            #### Forward pass ####
            # Hidden layer
            hidden_inputs = np.dot(X,self.weights_input_to_hidden) # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

            # Output layer
            final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
            final_outputs = final_inputs # signals from final output layer
            
            ### Backward pass ###

            # Output error
            error = y - final_outputs # Output layer error is the difference between desired target and actual output.
            
            output_error_term = error*1
            
            # Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(output_error_term,self.weights_hidden_to_output.T)
            
            # Backpropagated error terms.
            hidden_error_term = hidden_error*hidden_outputs*(1-hidden_outputs)

            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term*X[:,None]
            # Weight step (hidden to output)
            delta_weights_h_o += output_error_term*hidden_outputs[:,None]

        # Update the weights.
        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records # update input-to-hidden weights with gradient descent step
 
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Forward pass ####
        # Hidden layer
        hidden_inputs = np.dot(features,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # Output layer
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs
def MSE(y, Y):
    return np.mean((y-Y)**2)
import sys

### Set the hyperparameters here ###
iterations = 8000
learning_rate = 0.3
hidden_nodes = 13
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['waiting_time']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['waiting_time'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['waiting_time'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim(ymax=1.0)
# figure_fig = plt.gcf()  # 'get current figure'
# figure_fig.savefig('hidden_nodes_15.eps', format='eps', dpi=1000)
fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['waiting_time']
predictions = network.run(test_features).T*std + mean
ax.plot(predictions[0][0:100], label='Prediction')
ax.plot((test_targets['waiting_time']*std + mean).values[0:100], label='Data')
ax.set_xlim(right=100)
ax.legend()

dates_str = data_bus_35.ix[test_data.index]['date'] + ' ' +data_bus_35.ix[test_data.index]['time'].map(str)
dates = pd.to_datetime(dates_str[0:100])
dates = dates.apply(lambda d: d.strftime('%b %d %H:00'))
ax.set_xticks(np.arange(len(dates))[12::20])
_ = ax.set_xticklabels(dates[12::20], rotation=45)
# predictions[0][0:10]

data_pred = predictions * (predictions > 0)
ridership = 0.436 * data_pred - 5
ridership = np.ceil(ridership)
ridership = ridership * (ridership > 0)
ridership[0]
fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['waiting_time']
predictions = network.run(test_features).T*std + mean
ax.plot(ridership[0][0:100], label='Ridership_Pred')

ax.set_xlim(right=100)
ax.legend()

dates = pd.to_datetime(dates_str[0:100])
dates = dates.apply(lambda d: d.strftime('%b %d %H:00'))
ax.set_xticks(np.arange(len(dates))[12::20])
_ = ax.set_xticklabels(dates[12::20], rotation=45)
##变量命名后缀: _21_22_0表示21-22点第一段数据

##取21:00-22:00这一小时的数据
ridership_21_22_0 = ridership[0][-26:-11]
ridership_21_22_1 = ridership[0][-60:-33]

on_road_time_21_22_0 = bus_35['on_road_time'][-26:-11]
on_road_time_21_22_1 = bus_35['on_road_time'][-60:-33]

##取18:00-19:00这一小时的数据
ridership_18_19_0 = ridership[0][-352:-334]
ridership_18_19_1 = ridership[0][-400:-358]

on_road_time_18_19_0 = bus_35['on_road_time'][-352:-334]
on_road_time_18_19_1 = bus_35['on_road_time'][-400:-358]


#统计21:00-22:00人数与行车时间
total_num_21_22 = 0
total_on_road_time_21_22 = 0

for n in ridership_21_22_0:
    total_num_21_22 = total_num_21_22 + n

for d in on_road_time_21_22_0:
    total_on_road_time_21_22 = total_on_road_time_21_22 + d
    
for n in ridership_21_22_1:
    total_num_21_22 = total_num_21_22 + n

for d in on_road_time_21_22_1:
    total_on_road_time_21_22 = total_on_road_time_21_22 + d
    
#统计18:00-19:00人数与行车时间
total_num_18_19 = 0
total_on_road_time_18_19 = 0

for n in ridership_18_19_0:
    total_num_18_19 = total_num_18_19 + n

for d in on_road_time_18_19_0:
    total_on_road_time_18_19 = total_on_road_time_18_19 + d
    
for n in ridership_18_19_1:
    total_num_18_19 = total_num_18_19 + n

for d in on_road_time_18_19_1:
    total_on_road_time_18_19 = total_on_road_time_18_19 + d


################
## total_on_road_time: 总行车时间
## t_stop: 首末站停车时间
## cx: 满载量
## det: 满载率
## N : 线路配车数
## h_35 : 线路35发车时间
## c0 : 上下乘客权重
################

t_stop = 80
cx = 60
det = 0.98
c0 = 0.6

#21：00-22：00的发车间隔
N_21_22 = np.ceil((total_num_21_22) / (cx * det))
h_35_21_22 = np.ceil((total_on_road_time_21_22) / (N_21_22 * 60))

#18:00-19:00的发车间隔
N_18_19 = np.ceil((total_num_18_19) / (cx * det))
h_35_18_19 = np.ceil((total_on_road_time_18_19) / (N_18_19 * 60))

N_18_19
###优化后成本计算######
## z:运营总成本，z1:乘客出行成本，z2:企业运营成本
## a0:乘客出行成本权重， b0:企业运营成本权重
##
## a1:等车时间费用权重, b1:在车时间费用权重, y1:公交票价权重
## Iw,Id分别为初次候车时间价值、在车时间转换为乘客出行费用的系数
## h_35:线路35的发车间隔
## L_35:调度周期内线路35的运营里程
## V_35:为线路35的公交车辆运行平均速度
## t1:公交票价
##
## a2,b2,y2分别为油耗、驾驶员工资和车辆折旧占企业运营成本的权重值
## 
## Iz:将公交车辆行驶里程转化为公交运营费用的转换
## con:单车每小时耗电量
## sl:驾驶员工资
## sita:车辆折旧
###########

Iw = 2 
Id = 3
t1 = 1
L_35 = 18
V_35 = 30

a0 = 0.5
b0 = 0.5
a1 = 0.3
b1 = 0.3
y1 = 0.4
a2 = 0.4
b2 = 0.3
y2 = 0.3

Iz = 10
sl = 15
con = 5
sita = 25

# ##21:00-22:00成本
# z1_21_22 = a1 * Iw * total_num_21_22 * h_35_21_22  + b1 * Id * total_num_21_22 * L_35 / V_35 + y1 * t1
    
# z2_21_22 = N_21_22 *(a2 * Iz * L_35 * con + b2 * sl + y2 * sita)

# z_21_22 = a0 * z1 + b0 * z2

# ##18:00-19:00成本
# z1_18_19 = a1 * Iw * total_num_18_19 * h_35_18_19  + b1 * Id * total_num_18_19 * L_35 / V_35 + y1 * t1
    
# z2_18_19 = N_18_19 *(a2 * Iz * L_35 * con + b2 * sl + y2 * sita)

# z_18_19 = a0 * z1_18_19 + b0 * z2_18_19

# z1_18_19
###优化后成本计算######
## z:运营总成本，z1:乘客出行成本，z2:企业运营成本
## a0:乘客出行成本权重， b0:企业运营成本权重
##
## a1:等车时间费用权重, b1:在车时间费用权重, y1:公交票价权重
## Iw,Id分别为初次候车时间价值、在车时间转换为乘客出行费用的系数
## h_35:线路35的发车间隔
## L_35:调度周期内线路35的运营里程
## V_35:为线路35的公交车辆运行平均速度
## t1:公交票价
##
## a2,b2,y2分别为油耗、驾驶员工资和车辆折旧占企业运营成本的权重值
## 
## Iz:将公交车辆行驶里程转化为公交运营费用的转换
## con:单车每小时耗电量
## sl:驾驶员工资
## sita:车辆折旧
###########

Iw = 2 
Id = 3
t1 = 1
L_35 = 18
V_35 = 30

a0 = 0.5
b0 = 0.5
a1 = 0.3
b1 = 0.3
y1 = 0.4
a2 = 0.4
b2 = 0.3
y2 = 0.3

Iz = 10
sl = 15
con = 5
sita = 25

# ##21:00-22:00成本
# z1_21_22 = a1 * Iw * total_num_21_22 * h_35_21_22  + b1 * Id * total_num_21_22 * L_35 / V_35 + y1 * t1
    
# z2_21_22 = N_21_22 *(a2 * Iz * L_35 * con + b2 * sl + y2 * sita)

# z_21_22 = a0 * z1 + b0 * z2

# ##18:00-19:00成本
# z1_18_19 = a1 * Iw * total_num_18_19 * h_35_18_19  + b1 * Id * total_num_18_19 * L_35 / V_35 + y1 * t1
    
# z2_18_19 = N_18_19 *(a2 * Iz * L_35 * con + b2 * sl + y2 * sita)

# z_18_19 = a0 * z1_18_19 + b0 * z2_18_19

# z1_18_19
####优化后：成本计算公式方法2####
## W:公交运营成本
## P:乘客出行成本
## P1:乘客等车成本(其中h/2表示乘客平均等待时间)
## P2:乘客在车成本
## C1:乘客在站点等车损失的时间成本系数
## C2:乘客在途中损失的时间成本系数
## l:乘客平均里程
######################

C1 = 0.53
C2 = 0.08
l = 5

#18:00-19:00
W_18_19 = sita * L_35 * (total_on_road_time_18_19 / h_35_18_19 / 60)

P1_18_19 = C1 * total_num_18_19 * (h_35_18_19 / 2)
P2_18_19 = C2 * (l/V_35) * total_num_18_19
P_18_19 = P1_18_19 + P2_18_19

z_18_19 = a0 * W_18_19 + b0 * P_18_19

#21:00-22:00
W_21_22 = sita * L_35 * (total_on_road_time_21_22 / h_35_21_22 / 60)

P1_21_22 = C1 * total_num_21_22 * (h_35_21_22 / 2)
P2_21_22 = C2 * (l/V_35) * total_num_21_22
P_21_22 = P1_21_22 + P2_21_22

z_21_22 = a0 * W_21_22 + b0 * P_21_22
W_21_22
###优化前成本计算######
## z:运营总成本，z1:乘客出行成本，z2:企业运营成本
## a0:乘客出行成本权重， b0:企业运营成本权重
##
## a1:等车时间费用权重, b1:在车时间费用权重, y1:公交票价权重
## Iw,Id分别为初次候车时间价值、在车时间转换为乘客出行费用的系数
## h_35_orgin:线路35的发车间隔
## N_orgin: 优化前发车数量
## L_35:调度周期内线路35的运营里程
## V_35:为线路35的公交车辆运行平均速度
## t1:公交票价
##
## a2,b2,y2分别为油耗、驾驶员工资和车辆折旧占企业运营成本的权重值
## L:调度周期内所有公交线路的运营总里程
## Iz:将公交车辆行驶里程转化为公交运营费用的转换
## con:单车百公里燃油消耗量
## sl:驾驶员工资
## sita:车辆折旧
##
## r_num:实际估计乘客人数
## r0:上下车乘客权重
###########

N_orgin = 6
h_35_orgin = 7

Iw = 2 
Id = 3
t1 = 1
L_35 = 18
V_35 = 30

a0 = 0.5
b0 = 0.5
a1 = 0.3
b1 = 0.3
y1 = 0.4
a2 = 0.4
b2 = 0.3
y2 = 0.3

Iz = 12
sl = 15
con = 5
sita = 25


# ##21:00-22:00成本
# pre_z1_21_22 = a1 * Iw * total_num_21_22 * h_35_orgin + b1 * Id * total_num_21_22 * L_35 / V_35 + y1 * t1
    
# pre_z2_21_22 = N_orgin *(a2 * Iz * L_35 * con + b2 * sl + y2 * sita)

# pre_z_21_22 = a0 * pre_z1 + b0 * pre_z2

# ##18:00-19:00成本
# pre_z1_18_19 = a1 * Iw * total_num_18_19 * h_35_orgin + b1 * Id * total_num_18_19 * L_35 / V_35 + y1 * t1
    
# pre_z2_18_19 = N_orgin *(a2 * Iz * L_35 * con + b2 * sl + y2 * sita)

# pre_z_18_19 = a0 * pre_z1_18_19 + b0 * pre_z2_18_19

# pre_z1_18_19
####优化前：成本计算公式方法2####
## pre_W:公交运营成本
## pre_P:乘客出行成本
## pre_P1:乘客等车成本(其中h/2表示乘客平均等待时间)
## pre_P2:乘客在车成本
## C1:乘客在站点等车损失的时间成本系数
## C2:乘客在途中损失的时间成本系数
## l:乘客平均里程
######################

#18:00-19:00
pre_W_18_19 = sita * L_35 * (total_on_road_time_18_19 / h_35_orgin / 60)

pre_P1_18_19 = C1 * total_num_18_19 * (h_35_orgin / 2)
pre_P2_18_19 = C2 * (l/V_35) * total_num_18_19
pre_P_18_19 = pre_P1_18_19 + pre_P2_18_19

pre_z_18_19 = a0 * pre_W_18_19 + b0 * pre_P_18_19

#21:00-22:00
pre_W_21_22 = sita * L_35 * (total_on_road_time_21_22 / h_35_orgin / 60)

pre_P1_21_22 = C1 * total_num_21_22 * (h_35_orgin / 2)
pre_P2_21_22 = C2 * (l/V_35) * total_num_21_22
pre_P_21_22 = pre_P1_21_22 + pre_P2_21_22

pre_z_21_22 = a0 * pre_W_21_22 + b0 * pre_P_21_22
pre_W_21_22
#18:00-19:00成本对比
a_z1_18_19 = (P_18_19 - pre_P_18_19) / pre_P_18_19
a_z2_18_19 = (W_18_19 - pre_W_18_19) / pre_W_18_19
a_z_18_19 = (z_18_19 - pre_z_18_19) / pre_z_18_19

#21:00-22:00成本对比
a_z1_21_22 = (P_21_22 - pre_P_21_22) / pre_P_21_22 
a_z2_21_22 = (W_21_22 - pre_W_21_22) / pre_W_21_22
a_z_21_22 = (z_21_22 - pre_z_21_22) / pre_z_21_22

a_z_18_19
print("18:00-19:00成本对比："+ "\n优化后：  乘客出行成本：" + str(P_18_19)[:7] + "元  " + "公交运营成本：" 
      + str(W_18_19)[:7] + "元   "+ "综合成本：" + str(z_18_19)[:7] + "元"+ "\n优化前：  乘客出行成本：" 
      + str(pre_P_18_19)[:7] + "元  " + "公交运营成本：" + str(pre_W_18_19)[:7] + "元   " + "综合成本：" 
      + str(pre_z_18_19)[:7] + "元"+"\n对比：  乘客出行成本：" + str(a_z1_18_19 * 100)[:5] + "%  " + "公交运营成本：" 
      + str(a_z2_18_19 * 100)[:5] + "%   "+ "综合成本：" + str(a_z_18_19 * 100)[:5] + "%")
print("21:00-22:00成本对比："+ "\n优化后：  乘客出行成本：" + str(P_21_22)[:7] + "元  " + "公交运营成本：" 
      + str(W_21_22)[:7] + "元   "+ "综合成本：" + str(z_21_22)[:7] + "元"+ "\n优化前：  乘客出行成本：" 
      + str(pre_P_21_22)[:7] + "元  " + "公交运营成本：" + str(pre_W_21_22)[:7] + "元   " + "综合成本：" 
      + str(pre_z_21_22)[:7] + "元"+"\n对比：  乘客出行成本：" + str(a_z1_21_22 * 100)[:5] + "%  " + "公交运营成本：" 
      + str(a_z2_21_22 * 100)[:5] + "%   "+ "综合成本：" + str(a_z_21_22 * 100)[:5] + "%") 
