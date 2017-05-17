import csv
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from babeltrace import CTFWriter
import babeltrace
trace_collection = babeltrace.TraceCollection()
import sys
from demangler import demangle
# get the trace path from the first command line argument
trace_names=[];
trace_timestamps=[];
t=0;
with open('calltrace.out') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        name = demangle(row['func_name']);
        print(name)
        trace_names.append(name)
        t=t+1
        trace_timestamps.append(t)

x=[];y=[];count=0;
for name in trace_names:
    y.append(int(hashlib.sha224(bytes(name, 'utf-8')).hexdigest(),16))

for time in trace_timestamps:
    x.append(time)
#    if count > 1000:
#            break
#

colors = np.random.rand(count)
#area = np.pi * (15 * np.random.rand(count))**2  # 0 to 15 point radii
plt.scatter(x, y, c=colors, alpha=0.5)

#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

#plt.xlabel('Kernel Event ID')
#plt.ylabel('# of Calls of A Kernel Event')
#plt.show()
