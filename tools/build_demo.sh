#!/bin/bash
set -x
rm -r demo/sofalog 
cp -ar sofalog demo
rm demo/sofalog/sofa.pcap 
./bin/sofa clean --logdir demo/sofalog
ls -alh demo/sofalog
set +x

