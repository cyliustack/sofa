#!/bin/bash
set -x
rm -r demo/sofalog 
./bin/sofa stat "dd if=/dev/zero of=dummy.out bs=10M count=100"
cp -ar sofalog demo
rm demo/sofalog/sofa.pcap 
./bin/sofa clean --logdir demo/sofalog
ls -alh demo/sofalog
set +x

echo "Please do 'git add demo' to include new demo updates."

