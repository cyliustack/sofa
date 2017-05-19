# 1. Introduction
SOFA: Swarm of Functions Analysis  
Authors: All the contributors of SOFA

# 2. Prerequisite

## 2-1. Perf Installation 
### Debian/Ubuntu
`sudo apt-get install perf` 
### CentOS 7
`sudo yum install perf`

## 2-2. Perf Configuration
`su`  
`echo -1 /proc/sys/kernel/perf_event_paranoid`    
With the command above, you get raw access to kernel tracepoints (specifically, you can mmap the file created by perf_event_open, I don't know what the implications are).

# 3. SOFA Build and Installation 
1. git clone https://github.com/cyliustack/sofa
2. cd sofa 
3. make 
4. sudo make install

# 4. How To Use
## For Case 1
```
sofa ls -ah
potato 
```
## For Case 2
```
sofa --logdir=/tmp/sofalog-001 python -c "import tensorflow as tf; print(tf.__version__)"
potato --logdir=/tmp/sofalog-001
```
## Interactive and Visualization Result Provided by Potato:  
![Alt text](demo.png)

