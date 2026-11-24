# Copyright (c) 2014-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LBANN/ScaFFold.
#
# SPDX-License-Identifier: (Apache-2.0)

import logging
import math
import os
import socket
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.distributed as dist
from distconv import DCTensor, DistConvDDP, ParallelStrategy
from distconv import distconv as distconv_ops
from torch.distributed.tensor import Replicate, Shard

from ScaFFold.datagen.get_dataset import get_dataset
from ScaFFold.unet import UNet
from ScaFFold.unet.conv3d import FastConv3d, _halo_plan
from ScaFFold.unet.native_compile import bind_native_convolution
from ScaFFold.utils.config_utils import EXECUTION_MODES
from ScaFFold.utils.data_types import AMP_DTYPE
from ScaFFold.utils.distributed import (
    get_device,
    get_local_rank,
    get_local_size,
    get_world_rank,
    get_world_size,
    initialize_dist,
)
from ScaFFold.utils.perf_measure import (
    adiak_value,
    annotate,
    begin_code_region,
    end_code_region,
    get_torch_context,
)
from ScaFFold.utils.trainer import (
    PyTorchTrainer,
    compute_training_loss_and_dice,
)
from ScaFFold.utils.utils import set_seeds, setup_mpi_logger
from ScaFFold.viz import standard_viz


def configure_execution_mode(mode, *, log=None):
    """Apply a run's kernel preferences before constructing its model.

    Native modes disable handwritten Triton kernels and regional GroupNorm
    compilation. Triton modes and ``custom`` reload the environment-aware
    defaults, including hardware guards and fallbacks, so a preceding native
    run cannot leave its opt-outs active. Model compilation is configured
    separately by the resolved Config flags.
    """
    if mode not in EXECUTION_MODES:
        raise ValueError(
            f"execution_mode must be one of {EXECUTION_MODES}, got {mode!r}"
        )

    from ScaFFold.unet import conv3d, group_norm

    enabled = False if mode.startswith("native-") else None
    conv3d.set_conv_triton_enabled(enabled)
    group_norm.set_triton_enabled(enabled)
    group_norm.set_compile_enabled(enabled)
    if mode == "triton-compiled":
        if log is None:
            log = logging.getLogger(__name__)
        log.warning(
            "execution_mode=triton-compiled is experimental; selecting this "
            "mode does not extend supported Triton devices or configurations."
        )


def check_resource_utilization(log, rank, world_size):
    """Check that we are properly utilizing resources"""

    # CPU
    log.debug(f"rank {rank}, world_size {world_size}")
    log.debug(f"Number of Physical Cores: {psutil.cpu_count(logical=False)}")
    log.debug(f"Number of Logical Cores: {psutil.cpu_count(logical=True)}")
    # GPU
    if torch.cuda.is_available() and rank == 0:
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        log.debug(f"Number of GPUs available: {num_gpus}")

    if torch.cuda.is_available():
        this_gpu = torch.cuda.current_device()
        log.debug(f"rank {rank}: Details of GPU {this_gpu}:")
        log.debug(f"  Name: {torch.cuda.get_device_name(this_gpu)}")
        log.debug(f"  Capability: {torch.cuda.get_device_capability(this_gpu)}")
        log.debug(
            f"  Memory Allocated: {torch.cuda.memory_allocated(this_gpu) / 1024**2:.2f} MB"
        )
        log.debug(
            f"  Memory Cached: {torch.cuda.memory_reserved(this_gpu) / 1024**2:.2f} MB"
        )
        log.debug(
            f"  Total Memory: {torch.cuda.get_device_properties(this_gpu).total_memory / 1024**2:.2f} MB"
        )
        log.debug(f"  Device Properties: {torch.cuda.get_device_properties(this_gpu)}")


def wrap_model_ddp(model, device, ps):
    """Wrap a model in DistConvDDP pinned to the compute device.

    The device comes from get_device(), which is the single source of truth
    for device selection: under per-rank GPU masking every rank's only
    visible device is index 0 regardless of its local rank, so passing the
    local rank as a device index would address a nonexistent device. On CPU
    (e.g. gloo test runs) DDP requires device_ids/output_device of None.
    """
    ddp_device_ids = [device.index] if device.type == "cuda" else None
    ddp_output_device = device.index if device.type == "cuda" else None
    return DistConvDDP(
        model,
        parallel_strategy=ps,
        device_ids=ddp_device_ids,
        output_device=ddp_output_device,
    )


def _compile_backend(backend=None, *, device_type="cpu"):
    """Preserve CPU low-precision rounding in Inductor forward and backward."""
    from torch._dynamo.backends.registry import lookup_backend

    if device_type == "cpu" and (backend is None or backend == "inductor"):
        # BF16 halo-gradient additions must round at the same points as eager
        # operations. Pass this as a compiler option so lazy backward compilation
        # retains it too, without changing process-wide Inductor configuration.
        def cpu_inductor(graph, inputs):
            return torch._inductor.compile(
                graph, inputs, options={"emulate_precision_casts": True}
            )

        return cpu_inductor
    return lookup_backend("inductor" if backend is None else backend)


def _compile_training_forward(
    model_forward, model, parallel_strategy, amp_enabled, compile_kwargs, *, fullgraph
):
    """Compile CE/Dice with the model or after its separately compiled blocks."""
    spatial_mesh = parallel_strategy.device_mesh[parallel_strategy.distconv_dim_names]
    num_shards = parallel_strategy.num_shards
    device_type = next(model.parameters()).device.type

    def loss_forward(logits, labels, class_weights):
        return compute_training_loss_and_dice(
            logits, labels, class_weights, spatial_mesh, num_shards, device_type
        )

    local_loss = (
        loss_forward if fullgraph else torch.compile(loss_forward, **compile_kwargs)
    )

    def train_forward(images, labels, class_weights):
        with torch.autocast(device_type, dtype=AMP_DTYPE, enabled=amp_enabled):
            logits = model_forward(images)
        return local_loss(logits, labels, class_weights)

    return (
        torch.compile(train_forward, **compile_kwargs) if fullgraph else train_forward
    )


def compile_unsharded_forward(
    model,
    *,
    fullgraph=False,
    backend=None,
    compile_loss=False,
    parallel_strategy=None,
    amp_enabled=False,
):
    """Compile the local U-Net while preserving its DCTensor API.

    In block mode, the output convolution remains eager. In fullgraph mode it
    joins the U-Net graph. The DCTensor conversion pair stays outside the local
    graph. With ``compile_loss``, fullgraph training also includes CE and Dice,
    while block mode compiles the loss separately after the model blocks.
    Validation keeps the logits-returning forward. Spatial shard counts must all be one;
    config validates that.
    """
    if compile_loss and parallel_strategy is None:
        raise ValueError("compile_loss requires parallel_strategy")
    compile_kwargs = {"fullgraph": True, "dynamic": False, "recompile_limit": 64}
    compile_kwargs["backend"] = _compile_backend(
        backend, device_type=next(model.parameters()).device.type
    )
    if fullgraph:
        original_forward = model.forward
        local_forward = torch.compile(original_forward, **compile_kwargs)
    else:
        for block in (*model.down_list, *model.up_list[:-1]):
            block.forward = torch.compile(block.forward, **compile_kwargs)
        local_forward = model.forward

    local_train_forward = None
    if compile_loss:
        local_train_forward = _compile_training_forward(
            original_forward if fullgraph else local_forward,
            model,
            parallel_strategy,
            amp_enabled,
            compile_kwargs,
            fullgraph=fullgraph,
        )

    def forward_with_dctensor_boundary(input, labels=None, class_weights=None):
        is_dctensor = isinstance(input, DCTensor)
        local = distconv_ops._ToTensor.apply(input) if is_dctensor else input
        if labels is not None:
            if local_train_forward is None or class_weights is None:
                raise ValueError("compiled training loss and class weights required")
            local_labels = (
                distconv_ops._ToTensor.apply(labels)
                if isinstance(labels, DCTensor)
                else labels
            )
            return local_train_forward(local, local_labels.squeeze(1), class_weights)
        output = local_forward(local)
        if is_dctensor:
            return DCTensor.from_shard(output, input._parallel_strategy)
        return output

    model.forward = forward_with_dctensor_boundary


def compile_spatial_forward(
    model,
    parallel_strategy,
    *,
    backend=None,
    compile_loss=False,
    amp_enabled=False,
):
    """Compile native U-Net convolutions with DistConv's halo exchanges.

    The first DCTensor call samples every convolution's validated halo plan and
    checks that none selected a ScaFFold Triton convolution kernel. After that,
    DCTensor conversion stays at the model boundary. Native convolutions call
    DistConv's forward and backward halo helpers, whose communication remains
    an opaque operation inside the compiled local model.
    Sampling happens under no-grad, before the actual first forward, so it
    cannot change the autograd graph of a training step.
    With ``compile_loss``, calls with labels also compile the CE/Dice objective
    and its spatial reductions. Calls without labels still return logits.
    """
    original_forward = model.forward
    convolutions = [m for m in model.modules() if isinstance(m, FastConv3d)]
    compiled_forward = None
    compiled_train_forward = None
    compile_kwargs = {"fullgraph": True, "dynamic": False, "recompile_limit": 64}
    compile_kwargs["backend"] = _compile_backend(
        backend, device_type=next(model.parameters()).device.type
    )

    def prepare_route(input):
        plans = {}

        # DistConv's eager convolution caches halos by replacing the input
        # wrapper's inner tensor with a narrow view of a larger halo buffer.
        # Warm a separate wrapper so the real first compiled input keeps the
        # same strides as every later batch.
        warm_local = input._tensor.detach().clone(memory_format=torch.preserve_format)
        warm_input = DCTensor.from_shard(warm_local, parallel_strategy)
        warm_input._is_periodic = input._is_periodic
        warm_input._periodic_shard_padding = input._periodic_shard_padding

        def capture_plan(module, args):
            dc_input = args[0]
            if not isinstance(dc_input, DCTensor):
                raise TypeError("spatial route requires DCTensor convolution inputs")
            plan = _halo_plan(
                dc_input,
                parallel_strategy,
                dc_input._tensor,
                module.weight,
                module.stride,
                module.padding,
                module.dilation,
            )
            plans[module] = plan

        hooks = [m.register_forward_pre_hook(capture_plan) for m in convolutions]
        try:
            with torch.no_grad():
                original_forward(warm_input)
        finally:
            for hook in hooks:
                hook.remove()

        invalid = [m for m in convolutions if m not in plans or plans[m] is None]
        triton_convolutions = [m for m in convolutions if m._triton_ok]
        valid = torch.tensor(
            [not invalid, not triton_convolutions],
            device=input.device,
            dtype=torch.int32,
        )
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid[0].item():
            raise ValueError(
                "spatial fullgraph compilation needs a supported halo plan "
                f"at every convolution on every rank; invalid locally: {invalid}"
            )
        if not valid[1].item():
            raise ValueError(
                "native spatial compilation cannot include ScaFFold Triton "
                "convolutions; use execution_mode: native-compiled. Spatial "
                "Triton compilation is experimental and unsupported by this route."
            )
        for module in convolutions:
            bind_native_convolution(module, plans[module])

    def forward_with_dctensor_boundary(input, labels=None, class_weights=None):
        nonlocal compiled_forward, compiled_train_forward
        if not isinstance(input, DCTensor):
            raise TypeError("spatial compiled U-Net requires a DCTensor input")
        if labels is not None and (not compile_loss or class_weights is None):
            raise ValueError("compiled training loss and class weights required")
        if compiled_forward is None:
            prepare_route(input)
            compiled_forward = torch.compile(original_forward, **compile_kwargs)
            if compile_loss:
                compiled_train_forward = _compile_training_forward(
                    original_forward,
                    model,
                    parallel_strategy,
                    amp_enabled,
                    compile_kwargs,
                    fullgraph=True,
                )
        local = distconv_ops._ToTensor.apply(input)
        if labels is not None:
            local_labels = (
                distconv_ops._ToTensor.apply(labels)
                if isinstance(labels, DCTensor)
                else labels
            )
            return compiled_train_forward(local, local_labels.squeeze(1), class_weights)
        output = compiled_forward(local)
        return DCTensor.from_shard(output, input._parallel_strategy)

    model.forward = forward_with_dctensor_boundary


def export_profiler_trace(prof, config, log, rank, world_size, ranks_per_node):
    """Write this rank's chrome trace, reporting failures instead of raising.

    A profiling rank reaches this while the others are already heading for the
    barrier that precedes rank-0 post-processing, so an exception here does not
    just lose a trace: it kills this rank and leaves every other rank blocked
    in that barrier until the collective times out. Failures are real (a run
    with zero training batches never starts the profiler, and export_chrome_trace
    then raises; the trace can also fill the filesystem), so log them and let
    the job finish.

    The trace is written into ``config.run_dir`` so it lands with the rest of
    the run's artifacts instead of wherever the job happened to be launched
    from.

    Returns the path written, or None if the trace could not be written.
    """
    # Round up: with a partly-filled last node, flooring would report one node
    # too few (and a ranks_per_node larger than the job would report none).
    nodes = max(1, math.ceil(world_size / max(1, ranks_per_node)))
    tracename = (
        f"torch-{socket.gethostname()}-r{rank}"
        f"-N{nodes}-n{world_size}"
        f"-ps{config.problem_scale}-e{config.epochs}"
        f"-nipf{config.n_instances_used_per_fractal}-{int(time.time())}.json"
    )
    tracepath = Path(getattr(config, "run_dir", None) or os.getcwd()) / tracename
    try:
        prof.export_chrome_trace(str(tracepath))
    except Exception as e:
        log.error(
            "Could not write PyTorch trace '%s': %s: %s. Continuing so the "
            "run can finish; a run with zero profiled steps never starts the "
            "profiler and has no trace to export.",
            tracepath,
            type(e).__name__,
            e,
        )
        return None
    log.info("Wrote PyTorch trace '%s'", tracepath)
    return tracepath


@annotate()
def main(kwargs_dict: dict = {}):
    #
    # Setup
    #

    config = Namespace(**kwargs_dict)

    # Set logging options
    log = setup_mpi_logger(__file__, config.verbose)
    configure_execution_mode(config.execution_mode, log=log)
    log.info("Execution mode: %s", config.execution_mode)

    # Two seeds, two phases. Data generation below draws only on the dataset
    # seed, so anything it takes from the global RNGs is a function of the
    # dataset -- whose cache key holds the dataset seed alone. Training is
    # re-seeded from the training seed once the dataset is in place.
    set_seeds(config.dataset_seed)
    log.info(
        "Seeds: dataset_seed=%d (fractals, volumes, train/val split), "
        "training_seed=%d (model initialization, data order)",
        config.dataset_seed,
        config.training_seed,
    )

    # Get MPI information
    rank = get_world_rank(required=True)
    world_size = get_world_size(required=True)

    # Optionally enable additional determinism settings
    if config.more_determinism:
        log.info(
            "more_determinism TRUE -- enabling additional determinism settings to improve training reproducibility"
        )
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Default
        torch.backends.cudnn.benchmark = True

    # Initialize DDP. ScaFFold always runs the benchmark as a distributed job;
    # a one-rank launch is the supported singleton case.
    begin_code_region("init_ddp")
    if not dist.is_initialized():
        log.info("Initializing distributed process group...")
        initialize_dist(log, rendezvous="env")
    else:
        log.info("Distributed process group already initialized by launcher.")
    end_code_region("init_ddp")

    # More useful info
    log.debug(f"Host={socket.gethostname()} PID={os.getpid()}")
    log.debug(f"PyTorch {torch.__version__}, CUDA/ROCm {torch.version.cuda}")
    backend = dist.get_backend() if dist.is_initialized() else "none"
    log.debug(
        f"Backend={backend}, world_size={world_size}, rank={rank}, local_rank={get_local_rank()}"
    )
    log.info(f"rank={rank}, world_size={world_size}")

    total_distconv_shards = math.prod(config.dc_num_shards)
    if world_size % total_distconv_shards != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by total number of "
            f"distconv shards={total_distconv_shards}"
        )

    # Generate or retrieve dataset
    begin_code_region("get_dataset")
    dataset_dir = get_dataset(
        config, require_commit=config.dataset_reuse_enforce_commit_id
    )
    config.dataset_dir = dataset_dir
    end_code_region("get_dataset")

    # Everything from here on is training. Seeding it from the training seed
    # alone keeps model initialization and data order independent of the
    # dataset seed and of whether this run generated the dataset or reused a
    # cached one (datagen reseeds the Python and NumPy global RNGs).
    set_seeds(config.training_seed)

    # Initialize model
    begin_code_region("init_model")
    log.info(
        f"DistConv num_shards={config.dc_num_shards}, shard_dim={config.dc_shard_dims}"
    )
    device = get_device()
    log.info(f"Using device: {device}")
    model = UNet(
        n_channels=3,
        n_classes=config.n_categories + 1,
        trilinear=False,
        layers=config.unet_layers,
        group_norm_groups=config.group_norm_groups,
    )
    # DDP + DistConv setup
    ps = ParallelStrategy(
        num_shards=config.dc_num_shards,
        shard_dim=config.dc_shard_dims,
        device_type=device.type,
    )

    model = model.to(device, memory_format=torch.channels_last_3d)
    if config.activation_checkpointing:
        # Has to happen before the DDP wrap: afterwards the model is only
        # reachable as ``model.module``, and the wrapper does not forward the
        # method.
        log.info(
            "activation_checkpointing TRUE -- recomputing block activations in "
            "the backward pass instead of storing them"
        )
        model.use_checkpointing()
    if config.torch_compile:
        # Compiling the bound method keeps the model's state_dict keys intact,
        # so checkpoints remain compatible with uncompiled runs.
        if config.torch_compile_spatial:
            log.info(
                "torch_compile_spatial TRUE -- compiling the complete U-Net "
                "on local spatial shards with native convolutions and "
                "DistConv halo exchanges"
            )
            compile_spatial_forward(
                model,
                ps,
                compile_loss=config.torch_compile_loss,
                amp_enabled=config.torch_amp,
            )
        elif config.torch_compile_unsharded:
            if config.torch_compile_fullgraph:
                log.info(
                    "torch_compile_fullgraph TRUE -- compiling the complete "
                    "U-Net forward on local tensors"
                )
                if config.torch_compile_loss:
                    log.info(
                        "torch_compile_loss TRUE -- compiling the local U-Net "
                        "and training CE/Dice loss as one forward graph"
                    )
            else:
                log.info(
                    "torch_compile_unsharded TRUE -- compiling U-Net blocks "
                    "except the output convolution on local tensors"
                )
            compile_unsharded_forward(
                model,
                fullgraph=config.torch_compile_fullgraph,
                compile_loss=config.torch_compile_loss,
                parallel_strategy=ps if config.torch_compile_loss else None,
                amp_enabled=config.torch_amp,
            )
        else:
            raise ValueError(
                "torch_compile requires torch_compile_unsharded: 1 or "
                "torch_compile_spatial: 1; direct DCTensor compilation is unsupported"
            )
    # Wrap with DistConvDDP that corrects gradient scaling for dc submesh
    model = wrap_model_ddp(model, device, ps)
    # Store ps for use in the training loop
    config._parallel_strategy = ps
    end_code_region("init_model")

    check_resource_utilization(log, rank, world_size)

    #
    # Initialize trainer
    #
    if config.framework == "torch":
        # Optionally enable additional determinism settings
        if config.more_determinism:
            log.info(
                "Enabling additional determinism settings to improve training reproducibility"
            )
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        trainer = PyTorchTrainer(model, config, device, log)
        trainer.ps = ps
        trainer.spatial_mesh = ps.device_mesh[ps.distconv_dim_names]
        num_spatial_dims = len(ps.shard_dim)
        trainer.ddp_placements = [Shard(0)] + [Replicate()] * num_spatial_dims
        total_shards = math.prod(config.dc_num_shards)
        global_batch_size = config.local_batch_size * (world_size // total_shards)
        config.global_batch_size = global_batch_size
        ddp_ranks = world_size // total_shards
        adiak_value("global_batch_size", global_batch_size)
        adiak_value("ddp_ranks", ddp_ranks)
        adiak_value("total_shards", total_shards)
        adiak_value("num_spatial_dims", num_spatial_dims)
        if rank == 0:
            log.info(
                f"Effective global batch size = {global_batch_size} "
                f"(local_batch_size={config.local_batch_size} * "
                f"(world_size={world_size} / prod(dc_num_shards)={total_shards}))"
            )
            log.info(
                f"DDP ranks = {ddp_ranks} "
                f"world_size={world_size} // prod(dc_num_shards)={total_shards}"
            )
        too_small_splits = []
        if global_batch_size > trainer.n_train:
            too_small_splits.append(f"training n_train={trainer.n_train}")
        if global_batch_size > trainer.n_val:
            too_small_splits.append(f"validation n_val={trainer.n_val}")
        if too_small_splits:
            raise ValueError(
                "Effective global batch size exceeds available samples: "
                f"global_batch_size={global_batch_size}, "
                f"{', '.join(too_small_splits)}, "
                f"local_batch_size={config.local_batch_size}, "
                f"world_size={world_size}, "
                f"dc_num_shards={config.dc_num_shards}"
            )

    else:
        raise RuntimeError(
            "Invalid framework specified. Currently [torch] is the supported framework."
        )

    #
    # Run the training
    #
    ranks_per_node = get_local_size()
    prof_ctx, TORCH_PERF_LOCAL = get_torch_context(ranks_per_node, rank)
    with prof_ctx as prof:
        begin_code_region("cleanup_or_resume")
        trainer.cleanup_or_resume()
        end_code_region("cleanup_or_resume")
        begin_code_region("warmup")
        trainer.warmup()
        end_code_region("warmup")
        begin_code_region("train")
        trainer.train(profiler=prof if TORCH_PERF_LOCAL else None)
        end_code_region("train")
    if TORCH_PERF_LOCAL:
        export_profiler_trace(prof, config, log, rank, world_size, ranks_per_node)

    # Results are final here; synchronize before rank-0 post-processing so a
    # post-processing failure on rank 0 cannot strand the other ranks in a
    # collective they never reach.
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    #
    # Calculate benchmark score
    #
    if rank == 0:
        outfile_path = trainer.outfile_path
        train_data = np.genfromtxt(outfile_path, dtype=float, delimiter=",", names=True)
        if train_data.size == 0:
            raise RuntimeError(
                f"Training stats file '{outfile_path}' contains no epoch rows; "
                "cannot compute a benchmark score."
            )
        total_train_time = train_data["epoch_duration"].sum()
        if "total_optimizer_steps" in train_data.dtype.names:
            optimizer_steps = np.atleast_1d(train_data["total_optimizer_steps"])
            total_optimizer_steps = int(optimizer_steps[-1])
        elif "optimizer_steps" in train_data.dtype.names:
            total_optimizer_steps = int(
                np.atleast_1d(train_data["optimizer_steps"]).sum()
            )
        else:
            total_optimizer_steps = int(getattr(trainer, "total_optimizer_steps", 0))
        adiak_value("total_optimizer_steps", total_optimizer_steps)
        epochs = np.atleast_1d(train_data["epoch"])
        total_epochs = int(epochs[-1])
        if config.epochs == -1:
            fom = 1.0 / total_train_time
            fom_per_megasecond = fom * 1_000_000
            adiak_value("FOM", fom)
            log.info(
                f"FOM = {fom} (1 / total_train_time={total_train_time:.6f} seconds). "
                f"{fom_per_megasecond} (1/Megaseconds). "
                f"This FOM is specific to problem_scale={config.problem_scale}, "
                f"target_dice={config.target_dice}, "
                f"n_categories={config.n_categories}, "
                f"n_instances_used_per_fractal={config.n_instances_used_per_fractal}, "
                f"dataset_seed={config.dataset_seed}, "
                f"unet_bottleneck_dim={config.unet_bottleneck_dim}, "
                f"optimizer={config.optimizer}, "
                f"starting_learning_rate={config.starting_learning_rate}, "
                f"min_learning_rate={config.min_learning_rate}, "
                f"T_0={config.T_0}, T_mult={config.T_mult}, "
                f"disable_scheduler={config.disable_scheduler}, "
                f"dc_shard_dims={config.dc_shard_dims}."
            )
            extra_msg = f"Trained to >= {config.target_dice} validation dice score in {total_train_time:.2f} seconds, {total_epochs} epochs, {total_optimizer_steps} optimizer steps."
        else:
            extra_msg = f"Completed in {total_train_time:.2f} seconds, {total_epochs} epochs, {total_optimizer_steps} optimizer steps."

        log.info(
            f"Benchmark run at scale {config.problem_scale} complete. \n{extra_msg}"
        )

        # Generate plots
        log.info("Generating figures on rank 0...")
        begin_code_region("generate_figures")
        standard_viz.main(config)
        end_code_region("generate_figures")

    return 0
