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
    parser.add_argument('--gpus', help='list of numbers of GPUs to test, e.g. "1,2,4,8"', default='1')
    args = parser.parse_args()

    models = args.models.split(',')
    gpus = args.gpus.split(',')

    if args.clean:
        subprocess.call('rm t-bench-steptime-*.out', shell=True)

    if args.collect:
        for gpu in gpus:
            gpu = int(gpu) 
            for model in models:
                for i in range(0,args.num_runs):
                    filename = 't-bench-steptime-%s-gpu%d-%d.out' % (model, gpu, i)
                    print('Collection RUN-%d'%i) 
                    with open(filename,'w') as f:
                        subprocess.call('sudo sysctl -w vm.drop_caches=3', shell=True)
                        subprocess.call('~/workspace/scout/t-bench --model=%s --num_batches=20 --synthetic --num_gpus=%d' % (model, gpu), shell=True, stdout=f, stderr=f)
                    filename = 't-bench-steptime-withsofa-%s-gpu%d-%d.out' % (model, gpu, i)
                    print('Collection with SOFA RUN-%d'%i) 
                    with open(filename,'w') as f:
                        subprocess.call('sudo sysctl -w vm.drop_caches=3', shell=True)
                        subprocess.call('sofa record "~/workspace/scout/t-bench --model=%s --num_batches=20 --synthetic --num_gpus=%d" --logdir sofalog-%s-%d' % (model, gpu, model, i), shell=True, stdout=f, stderr=f)
        

    if args.report:
        data_models=[]
        xlabels = []
        for gpu in gpus:
            gpu = int(gpu)
            for model in models:
                data = []
                for i in range(0,args.num_runs):
                    filename = 't-bench-steptime-%s-gpu%d-%d.out'%(model, gpu, i)
                    #print('Report RUN-%d'%i) 
                    value1 = 0 
                    value2 = 0
                    with open(filename, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.find('total images/sec:') != -1:
                                value1 = 64.0/float(line.split()[2])
                                break
                    filename = 't-bench-steptime-withsofa-%s-gpu%d-%d.out'%(model, gpu, i)
                    with open(filename, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.find('total images/sec:') != -1:
                                value2 = 64.0/float(line.split()[2])
                                break
                    data.append(100*(value2-value1)/value1)
                data_models.append(data)
                xlabels.append(model+'-'+'G%d'%gpu)
        print(data_models)
        
        data = np.asarray(data_models).T
        fig, axs = plt.subplots(1, 1)
        axs.boxplot(data)
        axs.set_title('Overhead Evaluation (%)')
        axs.axhline(linewidth=1, color='b')
        axs.set_xticklabels(xlabels, rotation=45, fontsize=6)
        fig.savefig('report.png', dpi=900)
   
    
    print("Evaluation is done.")
