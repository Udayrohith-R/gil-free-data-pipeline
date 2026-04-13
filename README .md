# GIL-Free Data Ingestion Pipeline for LLM Training

> High-performance preference data loader that bypasses the Python Global Interpreter Lock (GIL) for RLHF training pipelines.

## The Problem

Standard PyTorch DataLoaders are bottlenecked by Python's GIL during tokenization and data preprocessing. When training large language models on preference data (for RLHF), GPUs sit idle waiting for the CPU to finish tokenizing the next batch. This wastes thousands of dollars in compute per training run.

## The Solution

A multi-layered approach to eliminate GIL contention:

```
┌─────────────────────────────────────────────────────┐
│                   Training Loop (GPU)                │
│                                                     │
│    ┌───────────────────────────────────────────┐    │
│    │      Shared Memory Ring Buffer             │    │
│    │      (Zero-Copy Transfer)                  │    │
│    └──────────────────┬────────────────────────┘    │
│                       │                             │
│    ┌──────────────────┴────────────────────────┐    │
│    │     Async Prefetch Queue (N batches)       │    │
│    └──────────────────┬────────────────────────┘    │
│                       │                             │
│    ┌──────────────────┴────────────────────────┐    │
│    │   GIL-Free Tokenizer (C/Rust Extension)   │    │
│    │   - Multi-threaded BPE (no GIL)           │    │
│    │   - Deterministic shuffle in C            │    │
│    │   - Direct memory writes                  │    │
│    └──────────────────┬────────────────────────┘    │
│                       │                             │
│    ┌──────────────────┴────────────────────────┐    │
│    │   Memory-Mapped Dataset (mmap)            │    │
│    │   - No full dataset load into RAM         │    │
│    │   - OS-level page caching                 │    │
│    └───────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## Key Optimizations

| Optimization | Standard PyTorch | GIL-Free Pipeline |
|---|---|---|
| Tokenization | Python (GIL-bound) | C/Rust threads (no GIL) |
| Data Transfer | pickle serialization | Shared memory (zero-copy) |
| Prefetching | DataLoader workers (IPC overhead) | Ring buffer (lock-free) |
| Shuffling | Python random (GIL) | C Fisher-Yates (no GIL) |
| Memory | Full dataset in RAM | Memory-mapped files |

## Architecture

### 1. C/Rust Tokenizer Extension
- BPE tokenization running in native threads, completely bypassing the GIL
- In production: `tiktoken-rs` via PyO3 for full compatibility with Claude's tokenizer
- Multi-threaded batch processing using pthreads/rayon

### 2. Shared Memory Ring Buffer
- Lock-free ring buffer using `multiprocessing.shared_memory`
- Zero serialization overhead between producer and consumer
- Configurable slot count for backpressure management

### 3. Async Prefetch Pipeline
- Background thread tokenizes N batches ahead of the training loop
- GPU never waits for tokenization to complete
- Automatic backpressure when consumer is slower than producer

### 4. Benchmark Harness
- Head-to-head comparison with standard Python DataLoader
- Metrics: tokens/sec, time-to-GPU, queue wait time, throughput

## Usage

```python
from gil_free_loader import GILFreePreferenceLoader

loader = GILFreePreferenceLoader(
    data_path="preference_data.jsonl",
    batch_size=32,
    max_seq_len=4096,
    num_workers=4,
    prefetch_batches=8,
)

for batch in loader:
    # batch.prompt_ids, batch.chosen_ids, batch.rejected_ids
    # Already tokenized, ready for GPU transfer
    loss = rlhf_train_step(batch)

# Performance metrics
print(loader.get_metrics())
```

## Benchmarking

```bash
python gil_free_loader.py
```

## Production Roadmap

- [ ] Rust tokenizer via PyO3 (replacing C extension)
- [ ] CUDA pinned memory for direct GPU DMA transfer
- [ ] Dynamic batch sizing based on sequence length
- [ ] Integration with PyTorch DistributedDataParallel
- [ ] Prometheus metrics endpoint for training pipeline monitoring

## Relevance

This project directly addresses the challenge of **eliminating Python GIL contention in LLM training code** — a specific infrastructure need for teams building large-scale RLHF training systems.

## Author

Uday — ML Infrastructure Engineer | Ex-Google DeepMind (Gemini)
