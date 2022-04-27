#!/bin/bash

########### CHANGE ME #########
bindir=~/build/mcpnet
datadir=~/src/mcpnet/data

# path to dataset
DS=gnw2000

for m in 0 1
do

    ${bindir}/bin/mi -i ${datadir}/${DS}.exp -o ${datadir}/${DS}.mi${m}.h5 -m ${m} -t 4

    ${bindir}/bin/auc_pr_roc -i ${datadir}/${DS}.mi${m}.h5 -x ${datadir}/${DS}_truenet.csv        

done
