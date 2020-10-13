import sys
import csv
import math
import numpy as np

in_name = sys.argv[1]
v_in = []
v_out = []
with open(in_name, 'r') as dat_file:
    next(dat_file) # skip v_in line
    reader = csv.reader(dat_file, delimiter=' ')
    tmp = True
    for row in reader:
        if row[0] == "v_out":
            tmp = False
        else:
            if tmp:
                v_in.append(row)
            else:
                v_out.append(row)
v_in = np.array(v_in, dtype=float)
v_out = np.array(v_out, dtype=float)
vals = []
print(v_in.shape)
for i in range(v_in.shape[0]):
    for j in range(v_in.shape[1]):
        if np.abs(v_out[i, j]) > 0.0001 and np.abs(v_in[i, j]) > 0.0001:
            vals.append(v_out[i, j] / v_in[i, j])

print("average eigenvalue {}".format(np.average(vals)))
print("standard deviation {}".format(np.std(vals)))
