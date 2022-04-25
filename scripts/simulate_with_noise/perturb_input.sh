#!/bin/bash

echo "It is recommended that this script be run in a python virtual environment."
echo "required packages:  numpy timeit "

######### CHANGE ME #######
srcdir=~/src/mcpnet
datadir=~/src/mcpnet/data
outdir_base=~/output
mkdir -p $outdir
perturb_script=scrpts/simulate_with_noise
logfile=perturb.log

##########################


for d in gnw2000
do

  # convert from exp text file to binary h5 file.
  datafile=${datadir}/${d}.h5
  outdir=$outdir_base}/${d}/perturbed
  # no noise set up.  we are not subsampling.

  # main comparison:  local noise = 20, global = 10, 10 runs of 150 samples each.
  # netbenchmark used 10 trials for method comparison and noise sensitivity
  for exp in $(seq 1 10)
  do
    outfile=${outdir}/${d}_l0.0_g0.0_x${exp}.h5
    echo "generating ${outfile}"
    ln -s $datafile $outfile

    outfile=${outdir}/${d}_l0.2_g0.1_x${exp}.h5
    echo "generating ${outfile}"

    python3 ${scripts_dir}/perturb.py -i $datafile -o $outfile \
      --local-method normal --local-noise-level 0.2 \
      --global-method lognormal --global-noise-level 0.1 \
      --seed $exp

  done


  # NOTE netbenchmark used 10 trials, each with 150 samples, with different noise levels.
  # for noise level comparison, global noise is default (0)
  for ln in 0.25 0.5 0.75 1.0
  do

    # netbenchmark used 10 trials for method comparison and noise sensitivity
    for exp in $(seq 1 10)
    do

      outfile=${outdir}/${d}_l${ln}_g0.0_x${exp}.h5
      echo "generating ${outfile}"

      python3 ${scripts_dir}/perturb.py -i $datafile -o $outfile \
        --local-method normal --local-noise-level $ln \
        --global-method lognormal --global-noise-level 0 \
        --seed $exp

    done
  done

  # for sample size experiment, default local (20 %, normal) and global noise (0)
done
