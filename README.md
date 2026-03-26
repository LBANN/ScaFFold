<div align='center'>
    <h1><b>ScaFFold Benchmark</b></h1>
    <p>A scalable deep learning benchmark: UNet trained on procedurally-generated, 3D fractal data</p>
</div>

## **About**

ScaFFold is the Scale-free Fractal benchmark for deep learning.

### Purpose

ScaFFold is a proxy application and benchmark representative of deep learning surrogate models that are trained on large, high-resolution, three-dimensional numerical simulations.
It is meant to support benchmarking at a variety of system scales and be adaptable to future deep learning systems innovations.
ScaFFold exercises much of the deep learning systems stack: I/O, compute, fine- and coarse-grained communication, and their integration in a framework.

### Characteristics

ScaFFold trains a 3D U-Net to perform semantic segmentation on a synthetic dataset composed of 3D volumes containing different classes of fractals.
The size of the problem is controlled by a *scale* parameter, which varies the size and complexity of the volumes and the depth of the U-Net.
The scale parameter is exponential: each increase roughly doubles the problem size; e.g., a scale 7 problem has a volume size of :math:`128^3` for each sample.
Using fractals enables large datasets to be generated in-situ (rather than distributed) while ensuring a complex yet tractable semantic segmentation problem.

The model is trained from a random initialization until convergence, which is defined to be a validation Dice score of at least 0.95.

## **Setup**

1. If running on an LLNL system, try using the scripts in `scripts/install-*.sh` for machine-specific install scripts.

1. Clone the repository:  
    `git clone https://github.com/LBANN/ScaFFold.git && cd ScaFFold`

1. Build the ccl plugin (if not using WCI wheel)
    `. scripts/install-rccl.sh`

1. Create and activate a python venv for running the benchmark:  
    `ml load python/3.11.5 && python3 -m venv .venvs/scaffoldvenv && source .venvs/scaffoldvenv/bin/activate && pip install --upgrade pip`

1. Necessary LLNL settings:
    - CUDA (matrix):
        1. `ml cuda/12.9.1 gcc/13.3.1 mvapich2/2.3.7`
        1. `export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH`
    - ROCm (elcap):
        1. `ml cce/21.0.0 cray-mpich/9.1.0 rocm/7.1.1 rccl/fast-env-slows-mpi`
            - If using WCI wheel:
                1. `export LD_PRELOAD=/opt/rocm-7.1.1/llvm/lib/libomp.so` # for libomp.so

1. Install the benchmark in the python venv:
    - CUDA: `pip install --no-binary=mpi4py .[cuda] --prefix=.venvs/scaffoldvenv --extra-index-url https://download.pytorch.org/whl/cu129 2>&1 | tee install.log`
    - ROCm (generic): `pip install --no-binary=mpi4py .[rocm] --prefix=.venvs/scaffoldvenv --extra-index-url https://download.pytorch.org/whl/rocm7.1 2>&1 | tee install.log`
    - ROCm (LLNL): `pip install .[rocmwci] --prefix=.venvs/scaffoldvenv 2>&1 | tee install.log`


## Running the benchmark

1. If running the benchmark for the first time, or running with different fractal parameters (`n_categories`, `variance_threshold`) than previously, generate fractal classes and instances:  
    `scaffold generate_fractals -c ScaFFold/configs/benchmark_default.yml`

    Note that the benchmark ships with an initial set of 50 fractal classes.

1. Once fractal generation completes, run the benchmark:  
    `torchrun-hpc -N 1 -n 4 --gpus-per-proc 1 $(which scaffold) benchmark -c ScaFFold/configs/benchmark_default.yml`

`benchmark` creates a folder for the benchmark run(s) at `base_run_dir` set in the config file. For reproducibility, we store a copy of the benchmark run config yml. Within each run subfolder, `benchmark` creates a yml config for that specific run.

After each run completes, statistics from the run are stored in `train_stats.csv`. Additionally, users can inspect plots of the training and validation losses over time in `<base_run_dir/figures`.

Parameters are set in a `.yml` config file and can be modified by the user:

```yml
# External/user-facing
base_run_dir: "benchmark_runs"     # Subfolder of $(pwd) in which to run jobs.
fract_base_dir: "fractals"         # Base directory for fractal IFS and instances.
n_categories: 5                    # Number of fractal categories present in the dataset.
n_instances_used_per_fractal: 145  # Number of unique instances to pull from each fractal class. There are 145 unique; exceeding this number will reuse some instances.
problem_scale: 6                   # Determines dataset resolution and number of unet layers. Default is 6.
unet_bottleneck_dim: 3             # Power of 2 of the unet bottleneck layer dimension. Default of 3 -> bottleneck layer of size 8.
seed: 42                           # Random seed.
batch_size: 1                      # Batch sizes for each vol size.
optimizer: "ADAM"                  # "ADAM" is preferred option, otherwise training defautls to RMSProp.

# Internal/dev use only
variance_threshold: 0.15           # Variance threshold for valid fractals. Default is 0.15.
n_fracts_per_vol: 3                # Number of fractals overlaid in each volume. Default is 3.
val_split: 25                      # In percent.
epochs: 100                        # Number of training epochs.
learning_rate: .0001               # Learning rate for training.
disable_scheduler: 1               # If 1, disable scheduler during training to use constant LR.
more_determinism: 0                # If 1, improve model training determinism.
datagen_from_scratch: 0            # If 1, delete existing fractals and instances, then regenerate from scratch.
train_from_scratch: 1              # If 1, delete existing train stats and checkpoint files. Keep 0 if want to restart runs where we left off.
dist: 1                            # If 1, use torch DDP.
torch_amp: 1                       # If 1, use mixed precision in training.
framework: "torch"                 # The DL framework to train with. Only valid option for now is "torch".
checkpoint_dir: "checkpoints"      # Subfolder in which to save training checkpoints.
checkpoint_interval: 1             # Number of epochs between saving training checkpoints.
loss_freq: 1                       # Number of epochs between logging the overall loss.
```

## How the benchmark works


### Overview

* 3D fractals, procedurally generated
* Train a UNet
    * Based on this work (link) which demonstrated that pretraining a UNet on synthetic fractals like this yielded signficant performance improvements


### Generating the fractal classes

Fractals are generated via an [Iterated Function System (IFS)](https://en.wikipedia.org/wiki/Iterated_function_system), which are composed of affine transformations. In our case, one affine transformation is determined by a set of 12 randomly generated parameters: the first 9 compose a 3x3 rotation matrix, and the last 3 compose a translation matrix:

$
F_{n+1} = \begin{pmatrix}
            a_n & b_n & c_n\\
            d_n & e_n & f_n\\
            g_n & h_n & i_n
            \end{pmatrix} F_n
            +
            \begin{pmatrix}
            j_n\\ k_n \\ l_n
            \end{pmatrix}
$,  
where $F_n$ is the $nth$ point in the fractal point cloud $F$.

<!-- # FIXME affine transformation visual/latex here -->

One IFS, which defines a fractal category, is a set of 2-4 affine transformations, plus an associated probability for each affine transformation to be chosen during the fractal generation process. The process for generating an IFS looks like the following:
```
For n  in n_categories :

1. Choose random number of affine transformations to comprise this fractal category (between 2 and 8)

2. For each affine transformation:

    1. Generate 12 random values for the transformation parameters. First 9 are rotation matrix, last 3 are translation

    2. Calculate det(rotation matrix), which determines the (unnormalized) probability of selecting this transformation when generating a fractal from this IFS

3. Normalize affine transformation probabilities -- at this point, we have an IFS for a specific fractal category.


Once we have an IFS, we must determine if this fractal category meets our variance criteria:

4. Generate a fractal point cloud from this IFS

5. Process the fractal point cloud (normalize, center, check for NaNs) and calculate variance

6. If var(fractal_point_cloud) < criteria (default 0.15): this fractal category is valid, so we save the above IFS params to csv
```

Below is an example of an IFS for a fractal with four affine transformations:
![image](ScaFFold/readme_figs/IFS_params.png)


### Generating fractals from each class

With our fractal categories determined, we then generate several fractals from each class. Each fractal instance of a given class is created by scaling one of the 12 IFS parameters, then generating the fractal as usual. We scale each parameter by values in [0.4, 1.6] in intervals of 0.1. This gives 12 unique variations for each of the 12 IFS parameters, plus the unscaled/base fractal, for a total of 145 fractal instances for each class. See examples of fractal instances for a specific class below:

![image](ScaFFold/readme_figs/fractal_variations_new.png)

The weights we use to scale IFS parameters look like the following:

![image](ScaFFold/readme_figs/IFS_param_weights.png)


### Dataset generation

Finally,  we are ready to generate a dataset for training our model. Each sample in the dataset is composed of several fractal instances (default=3), randomly selected from any category, overlain with eachother in a 3D voxel grid. Each fractal instance in a sample is placed in a random, non-centered location in the voxel grid if the `scale` parameter set in the sweep config file is <1; otherwise, each fractal instance is centered on the center of the voxel grid. Below is an outline of the data generation process:
```
For n  in n_volumes:

    1. Create volume  and mask  3D grids as matrix of 0s

    2. For fractal  in n_fractals_per_scene  (default=3):

        1. Pick a random fractal category from range(0, num_categories) 

        2. Pick a random instance of that fractal category and load that point cloud

        3. Project that fractal instance to a 3D voxel grid

        4. For xyz  in voxelgrid_coordinates :

            1. If instance  has points within voxel xyz:

                1. volume[xyz]   = [0, 0, 0.778] (make the voxel blue instead of black. the color is an arbitrary choice)

                2. mask[xyz]  = fractal_category 

    3. Save volume and mask  to files
```

### 1. Profiling with the PyTorch Profiler

Set `PROFILE_TORCH=ON` to generate a PyTorch profiling trace that can be read into [Perfetto](https://ui.perfetto.dev/).

### 2. Profiling with Caliper & Adiak

#### Building Caliper & Adiak with Python Bindings

##### A. Using Benchpark (via Spack)

1. Initialize experiment with Caliper
    - `benchpark system init --dest tuolumne llnl-elcapitan cluster=tuolumne`
    - `benchpark experiment init --dest scaffold --system tuolumne scaffold+rocm package_manager=spack-pip caliper=mpi,time,rocm`
1. `benchpark setup scaffold wkp`
1. `# Follow ramble instructions ...`

##### B. Manually

1. Activate python environment used to run the benchmark
    - `$ source /usr/workspace/mckinsey/ScaFFold-profiling/.venvs/scaffoldvenv/bin/activate`
1. Install required packages for profiling
    - `$ pip install ScaFFold[profiling]`
1. Build Adiak for metadata with `-DWITH_PYTHON_BINDINGS=ON`:
```
git clone https://github.com/LLNL/Adiak.git
cd Adiak && git submodule init && git submodule update
mkdir pybuild && cd pybuild
cmake -DENABLE_PYTHON_BINDINGS=ON -DENABLE_TESTS=OFF -DENABLE_MPI=ON -DCMAKE_INSTALL_PREFIX=. -Dpybind11_DIR=$(pybind11-config --cmakedir) ..
make && make install
```
4. Build Caliper with `-DWITH_PYTHON_BINDINGS=ON` (`ROCm`/`CUDA` flags depend on arch):
```
git clone https://github.com/LLNL/Caliper.git
cd Caliper
mkdir pybuild && cd pybuild
ml rocm/7.1.1
ml cuda/12.9.1
cmake -DWITH_PYTHON_BINDINGS=ON \
   -DWITH_ROCPROFILER=ON \
   -DWITH_CUPTI=ON \
   -DWITH_NVTX=ON \
   -DWITH_MPI=ON \
   -DWITH_ADIAK=ON \
   -Dadiak_DIR=/usr/workspace/mckinsey/Adiak/pybuild/lib/cmake/adiak/ \
   -Dpybind11_DIR=$(pybind11-config --cmakedir) \
   -DCMAKE_INSTALL_PREFIX=. ..
make && make install
```
5. Note: manual build requires manually exporting `pycaliper` and `pyadiak` into `PYTHONPATH` at runtime (using a spack environment would avoid this)
```
export PYTHONPATH=/usr/workspace/mckinsey/Caliper/pybuild-adiak/lib/python3.11/site-packages/:$PYTHONPATH
export PYTHONPATH=/usr/workspace/mckinsey/Adiak/pybuild/lib/python3.11/site-packages:$PYTHONPATH
# Avoid error with MPI profiling service
export LD_LIBRARY_PATH=/usr/workspace/mckinsey/ScaFFold-profiling-manual/.venvs/scaffoldvenv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
```

#### Profiling ScaFFold with Caliper

1. Use the `CALI_CONFIG` environment variable to select a Caliper profiling configuration. If this variable is not defined, the annotated regions will not do anything, other than a function call and if check.
    - `$ CALI_CONFIG="spot(output=test.cali,profile.mpi)" scaffold benchmark -c ScaFFold/configs/benchmark_default.yml -j`
