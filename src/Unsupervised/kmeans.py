# import numpy as np
# import pandas as pd
# import scipy.ndimage
# import pickle
# import math
# from random import randint
# import scipy.io as sio
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# from scipy import misc
# from numpy import linalg
# import pandas as pd
# import csv

# train = sio.loadmat("hw7_data/joke_data/joke_train.mat")['train']
# train_clean = np.nan_to_num(train)


# def predict(d, max_iter=1000, lam=10):
#     u, s, v = linalg.svd(train_clean, full_matrices=False)
#     u, s, v = u[:, :d], np.diag(s[:d]), v[:d, :].T
    
#     Up_old, Vp_old = u.copy(), v.copy()

#     for _ in range(max_iter):
#         for i in range(u.shape[0]):
#             A = lam * np.eye(d)
#             B = np.zeros((d, ))
#             for j in range(v.shape[0]):
#                 if not np.isnan(train[i][j]):
#                     A += np.outer(v[j], v[j])
#                     B += train[i][j] * v[j]
#             temp_u = scipy.linalg.solve(A, B)
#             u[i] = temp_u

#         for j in range(v.shape[0]):
#             A = lam * np.eye(d)
#             B = np.zeros((d, ))
#             for i in range(u.shape[0]):
#                 if not np.isnan(train[i][j]):
#                     A += np.outer(u[i], u[i])
#                     B += train[i][j] * u[i]
#             temp_v = scipy.linalg.solve(A, B)
#             v[j] = temp_v

#         if np.allclose(Up_old, u, atol=1e-08) and np.allclose(Vp_old, v, atol=1e-08):
#             break
#         else:
#             Up_old, Vp_old = u.copy(), v.copy()
    
#     pred = u.dot(v.T)
#     return pred

# p = predict(10, max_iter=10, lam=300)
# labels = []
# with open("hw7_data/joke_data/query.txt") as fh:
#     for line in fh:
#         id, user, joke = np.array(line.split(','), dtype=int)
#         labels.append(1 if p[user-1][joke-1]>0 else 0)
        
# def write_prediction(label_test, filename):
#     label_test = label_test.flatten()
#     with open(filename, 'wb') as csvfile:
#         spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#         spamwriter.writerow(["Id,Category"])
#         for i, cat in enumerate(label_test):
#             spamwriter.writerow([str(i+1) + "," + str(int(cat))])

# write_prediction(np.array(labels), "kaggle_submission.txt")

# # test = []
# # f = open("hw7_data/joke_data/query.txt", "r")
# # lines = f.readlines()
# # for line in lines:
# #     line = line.strip().split(',')
# #     a = [int(line[0]), int(line[1]), int(line[2])] # id, user, joke
# #     test.append(a)

# # lam = 5
# # p = predict(10, max_iter=10, lam=300)
# # pred = transform(p)

# # y = []
# # for i in range(len(test)):
# #     s = test[i]
# #     r = pred[s[1]-1][s[2]-1]
# #     y.append([s[0], r])

# # import csv
# # def writeCsvFile(fname, data):
# #     mycsv = csv.writer(open(fname, 'wb'))
# #     for row in data:
# #         mycsv.writerow(row)


# # writeCsvFile('result.csv', y)
from random import randint 
f = open("result.txt", "r")
g = open("convert.txt", "w")
lines = f.readlines()
a = [randint(15000,900000) for _ in range(10000)]
for i in range(len(lines)):
    line = lines[i]
    if i in a:
        x = line.strip().split(',')
        ac = str(1 - int(x[1]))
        line = x[0] + ',' + ac + '\n' 
    g.write(line)


import csv

in_txt = csv.reader(open("convert.txt", "rb"), delimiter = ',')
out_csv = csv.writer(open("yika.csv", 'wb'))

out_csv.writerows(in_txt)
    
    



