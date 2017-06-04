#!/bin/sh
ssh hpc0 iperf -s &
ssh hpc1 sleep 10 && pkill -x iperf &
ssh hpc1 iperf -c hpc1 &

