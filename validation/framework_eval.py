#!/usr/bin/env python3
import numpy as np
import pandas as pd
import subprocess
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='SOFA AISI Evaluation for Frameworks')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--num_runs', help='# of runs to obtain mean, std, etc..)', type=int, default=10)
    parser.add_argument('--models', help='list of models, e.g. "resnet50,vgg16"', default='resnet50,vgg16')
    args = parser.parse_args()


    models = args.models.split(',')

    if args.clean:
        subprocess.call('rm t-bench-steptime-*.out', shell=True)

    if args.collect:
        for model in models:
            for i in range(0,args.num_runs):
                filename = 't-bench-steptime-%s-%d.out'%(model,i)
                print('Collection RUN-%d'%i) 
                with open(filename,'w') as f:
                    subprocess.call('sudo sysctl -w vm.drop_caches=3', shell=True)
                    subprocess.call('sofa record "~/workspace/scout/t-bench --model=%s --num_batches=20 --synthetic --logdir sofalog-%s-%d"'%(model, model, i), shell=True, stdout=f, stderr=f)
        
    if args.report:
        data_models=[]
        for model in models:
            data = []
            for i in range(0,args.num_runs):
                filename = 't-bench-steptime-%d.out'%i
                #print('Report RUN-%d'%i) 
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.find('total images/sec:') != -1:
                            #print(line)
                            data.append(float(line.split()[2]))
            data_models.append(data)
        df = pd.DataFrame(data_models, columns=models)
        print(df.describe())
        fig = plt.figure()
        df.plot.box()
        plt.xlabel('models', fontsize=16)
        plt.ylabel("images/s", fontsize=16)
        fig.savefig("report.pdf")
   
    
    print("Evaluation is done.")
