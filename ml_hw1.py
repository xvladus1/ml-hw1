
# coding: utf-8

# In[83]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NUM_FEATURES = 7
LAMBDA_MAX = 5000

# features: 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year'
x_train = pd.read_csv('/Users/vscherbich/Desktop/ml/hw1-data/X_train.csv')
# sample
'''
0.30957,-0.36707,0.45545,-0.20083,-0.73992,-0.80885,1
0.30957,0.3592,-0.11611,-0.038361,0.16625,-0.80885,1
-0.86291,-0.99778,-0.89551,-1.2251,-0.55868,-0.26592,1
-0.86291,-0.69198,-0.42787,-0.56226,-0.15997,1.0914,1
-0.86291,-0.92133,-0.63571,-1.251,-0.41369,0.81993,1
'''

# features: 'mpg'
y_train = pd.read_csv('/Users/vscherbich/Desktop/ml/hw1-data/y_train.csv')
# sample
'''
-3.4459
-5.4459
5.5541
11.554
12.254
'''

# plot formatting
styles = ['ro', 'bo', 'go', 'yo', 'bo', 'mo', 'co']
labels = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'mpg']

# matrix stuff
XT = x_train.T

identity_matrix = np.identity(NUM_FEATURES)
XTY = np.dot(XT, y_train)
XTX = np.dot(XT, x_train)

print 'done'


# In[88]:

# (a)  For λ = 0, 1, 2, 3, . . . , 5000, solve for wRR.
for l in range(0, LAMBDA_MAX):
    # 1. Find y-axis values
    # w_rr = (l*I + X^T*X)^-1*X^T*y <==> w_rr = inv*X^T*y
    inv = np.linalg.inv(l*identity_matrix + XTX)
    
    w_rr = np.dot(inv, XTY)

    # 2. Find x-axis value (degree of freedom)
    # df(l) = trace[X*(l*I + X^T*X)^-1*X^T] <==> df(l) = trace[X*inv*X^T]
    dfl = np.trace(np.dot(
        np.dot(x_train, inv), XT)
    )
    
    for j in range(0, NUM_FEATURES):
        plt.plot(
            dfl,
            w_rr[j],
            styles[j],
            label=labels[j]
        )
    
plt.xlabel('df(lambda)')
plt.ylabel('degrees of freedom')
plt.legend(labels)

# render
plt.show()


# In[ ]:

# (b)  The 4th dimension (car weight) and 6th dimension (car year) clearly stand out over the other
#      dimensions. What information can we get from this?
Since y-axis represents mpg, we conclude two things:
    - car weight is inverserly proportional to mpg (fuel efficiency decreases as the weight increases)
    - car year affects mpg proportionally (later models tend to have better fuel efficiency)


# In[106]:

# (c) For λ = 0, . . . , 50, predict all 42 test cases.
LAMBDA_MAX_PRED = 50

x_test = pd.read_csv('/Users/vscherbich/Desktop/ml/hw1-data/X_test.csv')
y_test = pd.read_csv('/Users/vscherbich/Desktop/ml/hw1-data/y_test.csv')

# matrix stuff
XT_NEW = x_test.T

# for l in range(0, LAMBDA_MAX_PRED):
for l in range(0, 1):
    # 1. Find y-axis values
    # w_rr = (l*I + X^T*X)^-1*X^T*y <==> w_rr = inv*X^T*y
    inv = np.linalg.inv(l*identity_matrix + XTX)
    w_rr = np.dot(inv, XTY)
    print w_rr.shape
        
    y_pred = np.dot(XT_NEW, w_rr)
    print y_pred.shape
    
    rmse = np.sqrt(
        (1/42) * np.power((y_test - y_pred), 2)
    )
    print rmse.shape
    print rmse
    
    for j in range(0, NUM_FEATURES):
        plt.plot(
            l,
            rmse[j],
            styles[j],
            label=labels[j]
        )

