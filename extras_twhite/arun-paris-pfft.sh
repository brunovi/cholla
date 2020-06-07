#!/bin/bash

module load rocm
module load pfft
module load hdf5
module load gcc

OUTDIR="out.paris.pfft-amd.$(date +%m%d.%H%M%S)"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
export MV2_USE_CUDA=0
export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
export MV2_ENABLE_AFFINITY=0
srun -n4 -c16 -N1 --exclusive -p amdMI60 ../cholla.paris.pfft-amd ../parameter_file.txt |& tee tee
