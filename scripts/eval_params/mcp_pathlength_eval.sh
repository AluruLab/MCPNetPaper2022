#!/bin/bash


########### CHANGE ME #########
bindir=~/build/mcpnet
datadir=~/src/mcpnet/data

# path to dataset
DS=gnw2000

for m in 1 2  # MI methods
do

    ${bindir}/bin/mi -i ${datadir}/${DS}.exp -o ${datadir}/${DS}.mi${m}.h5 -m ${m} -t 4

    # check MPI's aupr
    echo "L = 1: AUPR"
    ${bindir}/bin/auc_pr_roc -i ${datadir}/${DS}.mi${m}.h5 -x ${datadir}/${DS}_truenet.csv | grep AUPR

    for diag in 0.0
    do

        ${bindir}/bin/diagonal -i ${datadir}/${DS}.mi${m}.h5 -o ${datadir}/${DS}.mi${m}.maxmin1.h5 --target-value $diag


        for i in {2..20}
        do
            first=$(( i/2 ))
            second=$(( (i+1)/2 ))
            prev=$(( i-1 ))

            echo $first $second $prev

            ${bindir}/bin/mcp -i ${datadir}/${DS}.mi${m}.h5 --first ${datadir}/${DS}.mi${m}.maxmin${first}.h5 --second ${datadir}/${DS}.mi${m}.maxmin${second}.h5 -o ${datadir}/${DS}.mi${m}.mcp${i}.h5 -x ${datadir}/${DS}.mi${m}.maxmin${i}.h5 -t 4 -m 5

            echo "L = $i, diag = ${diag}: AUPR"
            ${bindir}/bin/auc_pr_roc -i ${datadir}/${DS}.mi${m}.mcp${i}.h5 -x ~/data/${DS}/${DS}_truenet.csv | grep AUPR


        done

done

done
