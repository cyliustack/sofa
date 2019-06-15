#!/usr/bin/env python3
import subprocess
import time
import argparse

if __name__ == '__main__':
    bwa_is_recorded = False
    smb_is_recorded = False
    htvc_is_recorded = False

    parser = argparse.ArgumentParser(description='A SOFA wrapper which supports event-driven recording.')
    parser.add_argument('--trace-points', default='', metavar='Comma-sperated string list for interested keywords, e.g., "keyword1,keyword2"')
    args = parser.parse_args()

    while True:
        time.sleep(3)
        print(time.time())
        with open('/home/ubuntu/pbrun_error.log') as f:
            lines = f.readlines()
            lc = 0 
            for line in lines:
                #print('Line%d'%lc, line)
                lc = lc + 1
                if lc < 6:
                    continue
                if line.find('BWA') != -1 and not smb_is_recorded:
                    bwa_is_recorded = True
                    print('BWA begins at ', time.time()) 
                    time.sleep(120)
                    subprocess.call('sofa record "sleep 20" --profile_all_cpus --logdir=sofalog-bwa ', shell=True)
                    break
                if line.find('BQSR') != -1 and not smb_is_recorded:
                    smb_is_recorded = True
                    print('SMB begins at ', time.time()) 
                    time.sleep(120)
                    subprocess.call('sofa record "sleep 20" --profile_all_cpus --logdir=sofalog-smb ', shell=True)
                    break
                if line.find('HaplotypeCaller') != -1 and not htvc_is_recorded: 
                    htvc_is_recorded = True
                    print('HTVC begins at ', time.time()) 
                    time.sleep(120)
                    subprocess.call('sofa record "sleep 20" --profile_all_cpus --logdir=sofalog-htvc ', shell=True)
                    break
        if bwa_is_recorded and smb_is_recorded and htvc_is_recorded:
            print("Tracing is done.") 
            break
