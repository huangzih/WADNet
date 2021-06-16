import andi
import numpy as np
import pandas as pd
import argparse
import gc

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--N', type=int)
arg('--l', type=int)
args = parser.parse_args()

N = args.N
l = args.l
filename = './origin_data/data-3d-{}.csv'.format(l)
output = './pp_data_3d/data-3d-{}-pp.csv'.format(l)

AD = andi.andi_datasets()
X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=N, max_T=l+1, min_T=l, tasks=1, dimensions=3)

with open(filename, 'w') as f:
    f.write('pos;label\n')
    for i in range(len(X1[2])):
        f.write(','.join([str(j) for j in X1[2][i]]))
        f.write(';'+str(Y1[2][i])+'\n')
    f.close()

del X1, Y1
gc.collect()

data = pd.read_csv(filename, sep=';')
data['length'] = data['pos'].apply(lambda x: round(len(x.split(','))/3))

data['pos_x'] = data.apply(lambda x: ','.join(x['pos'].split(',')[:x['length']]), axis=1)
data['pos_y'] = data.apply(lambda x: ','.join(x['pos'].split(',')[x['length']:2*x['length']]), axis=1)
data['pos_z'] = data.apply(lambda x: ','.join(x['pos'].split(',')[2*x['length']:]), axis=1)

del data['pos']

def normalize(x):
    data = np.array([float(i) for i in x.split(',')])
    mean = np.mean(data)
    std = np.std(data)
    data2 = (data - mean)/std
    return ','.join([str(i) for i in data2])

data['pos_x'] = data['pos_x'].apply(lambda x: normalize(x))
data['pos_y'] = data['pos_y'].apply(lambda x: normalize(x))
data['pos_z'] = data['pos_z'].apply(lambda x: normalize(x))

data[['pos_x','pos_y','pos_z','length','label']].to_csv(output, index=False, sep=';')
