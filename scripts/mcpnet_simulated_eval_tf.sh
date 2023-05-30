#!/bin/bash
#!/bin/bash -x
#SBATCH --job-name=mcpnet_noise
#SBATCH --output=log/mcpnet/log_%J.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH --partition=shared

#### each task is 1 mpi process.
#### for MPI, ntasks is total MPI procs, 1 mpi per core, x per socket
#### for OMP, ntasks is 1, cpus-per-task = num threads, and set OMP_NUM_THREADS.
#### for this, it's a mix, so try ntasks = 4, cpu-bind=socket, cpus-per-task=16, ntasks-per-socket=1, OMP_NUM_THREADS=16

############# CHANGE ME ###########
bindir=~/build/mcpnet
srcdir=~/src/mcpnet
datadir_base=~/output
logfile=simulated.log
coeffs=${srcdir}/data/combos.csv
tf_file=${datadir_base}/yeast/infergs_yeast_tfs.txt
###############

date
pwd

export MPIEXEC_PREFIX_DEFAULT=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

function run_mcp {
    _bindir=$1
    _datadir=$2
    _datafile=$3
    _dataprefix=$4
    _coefffile=$5
    _truthfile=$6
    _logfile=$7
    _threads=$8

    # compute mi
    cmd="${_bindir}/bin/mcpnet -i ${_datafile} -o ${_datadir}/${_dataprefix} -m 1 2 3 4 -f ${_coefffile} -x ${_truthfile} -t ${_threads} --tf-input ${tf_file}"

    echo "$cmd"
    echo "$cmd" >> ${_logfile}
    eval "/usr/bin/time -v $cmd >> ${_logfile} 2>&1"

    rm ${_datadir}/${_dataprefix}.*.h5

}

threads=$SLURM_CPUS_PER_TASK

echo "test synthetic network" > $logfile

for percent in 100
do

for d in gnw2000
do
    truth=${src_dir}/data/${d}_truenet.csv
    datadir=${datadir_base}/${d}/perturbed
    echo "SAVING TO ${datadir}/${d}.aupr${percent}.csv"
    
    # noise free run 

    l=0.0
    g=0.0
    x=1
    dataprefix=${d}_l${l}_g${g}_x${x}_tf.run
    # process l 0.0 g 0.0
    datafile=${datadir}/${d}_l${l}_g${g}_x${x}.h5

    # clean up
   # rm ${datadir}/${dataprefix}.*.run.h5

    run_mcp $bindir $datadir $datafile $dataprefix $coeffs $truth $logfile $threads

    # default noisy run
    l=0.2
    g=0.1
    for x in 1 2 3 4 5 6 7 8 9 10
    do

        dataprefix=${d}_l${l}_g${g}_x${x}_tf.run
        # process l 0.2 g 0.1
        datafile=${datadir}/${d}_l${l}_g${g}_x${x}.h5

        # clean up
        # rm ${datadir}/${dataprefix}.*.run.h5

        run_mcp $bindir $datadir $datafile $dataprefix $coeffs $truth $logfile $threads

    done

    # testing different noise levels.
    g=0.0
    for l in 0.25 0.5 0.75 1.0
    do

        for x in 1 2 3 4 5 6 7 8 9 10
        do
    
            dataprefix=${d}_l${l}_g${g}_x${x}_tf.run
            # process l 0.2 g 0.1
            datafile=${datadir}/${d}_l${l}_g${g}_x${x}.h5

            # clean up
        #    rm ${datadir}/${dataprefix}.*.run.h5

            run_mcp $bindir $datadir $datafile $dataprefix $coeffs $truth $logfile $threads

        done
    done


done

done
