#!/bin/bash
module load gcc cuda fftw hdf5 python

export GPU_MPI="-DGPU_MPI"
export F_OFFLOAD="-qsmp=omp -qoffload"

export CHOLLA_ENVSET=1
