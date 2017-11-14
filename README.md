# 1. Introduction
SOFA: Swarm of Functions Analysis  
Authors: All the contributors of SOFA

# 2. Prerequisite

## 2-1. Installation 
`./tools/prerequisite.sh`  
`./tools/empower-tcpdump.sh $(whoami)`  
`echo "re-login to apply changes" && exit`
Simple Test:  
`tcpdump -w sofa.pcap`  
`tcpdump -r sofa.pcap`  
 
## 2-2. Perf Configuration
Let ormal users get raw access to kernel tracepoints:  
`sudo sysctl -w kernel.perf_event_paranoid=-1`  
Check the configuration result:  
`cat /proc/sys/kernel/perf_event_paranoid`  


# 3. Build Dependent Third-party Library
`wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz`
`tar -xvf 3.3.4.tar.gz && cd eigen-eigen-5a0156e40feb && mkdir build && cd build && cmake .. && make && sudo make install` 

# 4. SOFA Build and Installation 
1. git clone https://github.com/cyliustack/sofa
2. cd sofa 
3. CC=gcc CXX=g++ make 
4. sudo make install

# 5. How To Use
## For Case 1
```
cp examples/conf/default-single.cfg  .
make run
```
## For Case 2
```
cp examples/conf/default-single.cfg .
sofa --config default.cfg ls -ah  
potato .    
```
## For Case 3
```
cp examples/conf/default-cluster.cfg .
cp examples/start-all-example.sh .
Modify start-all-example.sh for names of involved nodes
sofa-dist "node0 node1" 10 "sh start-all-processes-on-all-nodes.sh" 
potato .    
```

## Interactive and Visualization Result Provided by Potato:  
![Alt text](./figures/demo.png)
![Alt text](./figures/demo2.png)
![Alt text](./figures/demo3.png)
![Alt text](./figures/demo4.png)





