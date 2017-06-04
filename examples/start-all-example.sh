#!/bin/sh
ssh hpc1 iperf -s &
ssh hpc2 iperf -c hpc1 &

