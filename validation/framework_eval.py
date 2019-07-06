#!/usr/bin/env python3
import numpy as np
import pandas as pd
import subprocess
import argparse
import matplotlib.pyplot as plt
from scipy import stats

def autolabel(axs, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        axs.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='SOFA AISI Evaluation for Frameworks')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--num_runs', help='# of runs to obtain mean, std, etc..)', type=int, default=10)
    parser.add_argument('--frameworks', help='list of frameworks, e.g. "tensorflow,pytorch"', default='tensorflow')
    parser.add_argument('--models', help='list of models, e.g. "resnet50,vgg16"', default='resnet50,vgg16')
    parser.add_argument('--gpus', help='list of numbers of GPUs to test, e.g. "1,2,4,8"', default='1')
    args = parser.parse_args()

    models = args.models.split(',')
    gpus = args.gpus.split(',')

    if args.clean:
        subprocess.call('rm t-bench-steptime-*.out', shell=True)

    if args.collect and 'tensorflow' in args.frameworks:
        for gpu in gpus:
            gpu = int(gpu) 
            for model in models:
                for i in range(0,args.num_runs):
                    filename = 't-bench-steptime-%s-gpu%d-%d.out' % (model, gpu, i)
                    print('tensorflow collection for model %s with GPUx%d RUN-%d' % (model, gpu, i)) 
                    with open(filename,'w') as f:
                        subprocess.call('sudo sysctl -w vm.drop_caches=3', shell=True)
                        subprocess.call('export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH; /home/ubuntu/workspace/scout/t-bench --model=%s --num_batches=20 --data_dir=/mnt/tmpfs/mini-imagenet/ --num_gpus=%d --batch_size=64 --strategy=parameter_server' % (model, gpu), shell=True, stdout=f, stderr=f)
                    filename = 't-bench-steptime-withsofa-%s-gpu%d-%d.out' % (model, gpu, i)
                    print('tensorflow+SOFA collection for model %s with GPUx%d RUN-%d' % (model, gpu, i)) 
                    with open(filename,'w') as f:
                        subprocess.call('sudo sysctl -w vm.drop_caches=3', shell=True)
                        subprocess.call('export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH; sofa record "/home/ubuntu/workspace/scout/t-bench --model=%s --num_batches=20 --data_dir=/mnt/tmpfs/mini-imagenet/ --num_gpus=%d --batch_size=64 --strategy=parameter_server"' % (model, gpu), shell=True, stdout=f, stderr=f)
 
    if args.collect and 'pytorch' in args.frameworks:
        print(gpus)
        for gpu in gpus:
            gpu = int(gpu)
            if gpu == 1:
                gpuids = '0'
            elif gpu == 2:
                gpuids = '0,1'
            elif gpu == 4:
                gpuids = '0,1,4,5'
            elif gpu == 8:
                gpuids = '0,1,2,3,4,5,6,7'
            else:
                gpuids= '0'

            batch_size = 16*gpu 
            for model in models:
                for i in range(0,args.num_runs):
                    filename = 'p-bench-steptime-%s-gpu%d-%d.out' % (model, gpu, i)
                    print('pytorch collection for model %s with GPUx%d RUN-%d' % (model, gpu, i))
                    with open(filename,'w') as f:
                        subprocess.call('sudo sysctl -w vm.drop_caches=3', shell=True)
                        subprocess.call('export CUDA_VISIBLE_DEVICES=%s; python ~/workspace/scout/pytorch_examples/imagenet/main.py -a %s /mnt/tmpfs/mini-imagenet/raw-data --epochs=1 --batch-size=%d' % (gpuids, model, batch_size), shell=True, stdout=f, stderr=f)
                    filename = 'p-bench-steptime-withsofa-%s-gpu%d-%d.out' % (model, gpu, i)
                    print('pytorch+SOFA collection for model %s with GPUx%d RUN-%d' % (model, gpu, i)) 
                    with open(filename,'w') as f:
                        subprocess.call('sudo sysctl -w vm.drop_caches=3', shell=True)
                        subprocess.call('export CUDA_VISIBLE_DEVICES=%s; sofa record "python ~/workspace/scout/pytorch_examples/imagenet/main.py -a %s /mnt/tmpfs/mini-imagenet/raw-data --epochs=1 --batch-size=%d"' % (gpuids, model, batch_size), shell=True, stdout=f, stderr=f)
        

    if args.report:
        xlabels = []
        means_before=[]
        means_after=[]
        stds_before=[]
        stds_after=[]
        if 'tensorflow' in args.frameworks:
            for gpu in gpus:
                gpu = int(gpu)
                for model in models:
                    befores=[]
                    afters=[]
                    for i in range(0,args.num_runs):
                        filename = 't-bench-steptime-%s-gpu%d-%d.out'%(model, gpu, i)
                        #print('Report RUN-%d'%i) 
                        with open(filename, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.find('total images/sec:') != -1:
                                    befores.append(gpu*64.0/float(line.split()[2]))
                                    break
                        filename = 't-bench-steptime-withsofa-%s-gpu%d-%d.out'%(model, gpu, i)
                        with open(filename, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.find('total images/sec:') != -1:
                                    afters.append(gpu*64.0/float(line.split()[2]))
                                    break
                    #data_models.append(data)
                    #xlabels.append(model+'-'+'G%d'%gpu)
                    means_before.append(np.mean(np.array(befores)))
                    stds_before.append(np.std(np.array(befores)))
                    means_after.append(np.mean(np.array(afters)))
                    stds_after.append(np.std(np.array(afters)))
                    xlabels.append(model+'-'+'G%d'%gpu)
                    #xlabels.append(model+'-'+'G%d'%gpu+'+SOFA')

        if 'pytorch' in args.frameworks:
            for gpu in gpus:
                gpu = int(gpu)
                if gpu == 8:
                    break
                for model in models:
                    befores=[]
                    afters=[]
                    for i in range(0,args.num_runs):
                        filename = 'p-bench-steptime-%s-gpu%d-%d.out'%(model, gpu, i)
                        with open(filename, 'r') as f:
                            lines = f.readlines()
                            values = []
                            for line in lines:
                                if line.find('Epoch: [0][') != -1:
                                    values.append(float(line.split('Time')[1].split()[0]))
                            befores.append(np.mean(np.asarray(values[1:-2])))
                        filename = 'p-bench-steptime-withsofa-%s-gpu%d-%d.out'%(model, gpu, i)
                        with open(filename, 'r') as f:
                            lines = f.readlines()
                            values=[]
                            for line in lines:
                                if line.find('Epoch: [0][') != -1:
                                    values.append(float(line.split('Time')[1].split()[0]))
                            afters.append( np.mean(np.asarray(values[1:-2]))) 
                        #data.append(100*(value2-value1)/value1)
                    means_before.append(np.mean(befores))
                    stds_before.append(np.std(befores))
                    means_after.append(np.mean(afters))
                    stds_after.append(np.std(afters))
                    xlabels.append(model+'-'+'G%d'%gpu)
                    #xlabels.append(model+'-'+'G%d'%gpu+'+SOFA')
        
        fig, axs = plt.subplots(1, 1)
        ind = np.arange(1,len(means_before)+1)  # the x locations for the groups
        print(ind)
        width = 0.25  # the width of the bars
        rects1 = axs.bar(ind - width/2, means_before, width, yerr=stds_before, label='no-SOFA')
        rects2 = axs.bar(ind + width/2, means_after, width, yerr=stds_after, label='with-SOFA')
        axs.set_title('Overhead Evaluation')
        axs.set_ylabel('step time (s)')
        #autolabel(axs, rects1, "left")
        #autolabel(axs, rects2, "right") 
        axs.axhline(linewidth=1, color='b')
        axs.set_xticklabels(xlabels, rotation=25, fontsize=7)
        start, end = axs.get_xlim()
        axs.xaxis.set_ticks(np.arange(start, end, 1))
        fig.savefig('report.png', dpi=900)
        a = np.array(means_before)
        b = np.array(means_after)
        tt,p = stats.ttest_rel(a,b)
        print(np.subtract(a,b))
        print(np.mean(np.abs(np.divide(np.subtract(b,a),a))))
        print(np.std(np.abs(np.divide(np.subtract(b,a),a))))
        print('two-tailed p-value: ', p)
    print("Evaluation is done.")
