#!/bin/bash

# Exit if target directory already exists
if [ -d "aws-ofi-nccl.git" ]; then
  echo "Directory 'aws-ofi-nccl.git' already exists. Exiting to avoid overwrite."
  return 1 2>/dev/null || exit 1
fi

rocm_version=7.1.0

module swap PrgEnv-cray PrgEnv-gnu
module load rocm/$rocm_version

git clone --recursive --branch v1.18.0 https://github.com/aws/aws-ofi-nccl.git aws-ofi-nccl.git

cd aws-ofi-nccl.git

installdir=$(pwd)/install

./autogen.sh

export LD_LIBRARY_PATH=$PWD/../rccl/install/lib:/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH

#CC=hipcc CXX=hipcc CFLAGS=-I$PWD/../rccl/install/include/rccl ./configure \
./configure \
  --with-libfabric=/opt/cray/libfabric/2.1 \
  --with-rocm=$ROCM_PATH \
  --prefix=$installdir

make
make install