# Introduction
SOFA : Swarm of Functions Analysis
Author Cheng-Yueh Liu

# Prerequisite

## Debian/Ubuntu
`sudo apt-get install perf nodejs` 

## CentOS 7
`sudo yum install perf nodejs`

# SOFA Installation 
1. git clone https://github.com/cyliustack/sofa
2. cd sofa 
3. make 
4. sudo make install

# How To Use
## For Case 1
```
sofa ls
potato 
```
## For Case 2
```
sofa --logdir=/tmp/sofalog-001 python -c "import tensorflow as tf; print(tf.__version__)"
potato --logdir=/tmp/sofalog-001
```
