# Uncertainty Benchmarking

## Overview

This repository contains the benchmarking scripts for my Master's thesis, "Schrödinger’s Carbon: Until Measured, Operational Emissions Remain Uncertain." The thesis investigates the deployment, spatiotemporal, and computational profile variables that affect uncertainty in the operational carbon footprint estimation of artificial intelligent inference tasks.

`Operational Carbon (gCO2eq) = Energy Consumed (kWh) * Carbon Intensity Factor (gCO2eq/kWh)`

To generate the dataset for the energy consumption factor in the equation above, inference benchmarks were run on four different models:

1. ResNet50 (Image Classification)
2. BERT-Large (Language Processing)
3. SDXL-Turbo (Text-to-Image)
4. GIT-Large Coco (Image Captioning)

Models were chosen to reflect a diverse array of task types. ResNet50 and BERT-Large were ran using MLPerf benchmarks[1]. SDXL-Turbo and GIT-Large-Coco were models chosen from HuggingFace and benchmarked with the dataset used by AI Energy Score[2].

This repository contains the code used to execute and monitor the GPU power usage for each model. Power measurements were recorded using `nvidia-smi` [3].

# Repository Structure

```
uncertainty_benchmarking/
├── main.py                # Entry point to run benchmarking
│                            # Contains MLPerf Benchmarking commands for ResNet50 and BERT-Large
├── huggingface/           # Benchmarking for HuggingFace Models
│   ├── sdxl-turbo.py
│   └── git-large-coco.py
├── installation.sh        # Installation script required for benchmarking
└── process_logs.py        # Script to extract timestamps and energy consumption from power
```

## Setup and Installation

### Prerequisites

The experiments were conducted on 12 NVIDIA RTX6000 nodes on CHI@UC cluster on Chameleon Cloud[4]. All nodes ran Ubuntu 22.04 LTS with CUDA 12.0. If running on a different setup, adjustments will need to be to made to `installation.sh`. Python 3.6^ is required to run benchmarking scripts.

1. Clone the repository.

```
git clone git@github.com:julxia/uncertainty_benchmarking.git
cd uncertainty_benchmarking
```

2. Make `installation.sh` executable and run installation

```
chmod+x installation.sh
installation.sh
```

3. Activate virtual environment

```
source mlc/bin/activate
```

4. Run benchmarking with required flags.

- `--server_id`: Server ID, used for log identification only (default: 0)
- `--model`: Comma-separated list of models to benchmark [`resnet50`, `bert-large`, `sdxl-turbo`, `glc`]
- `--iterations`: Number of iterations (default: 1)

**Sample commands**

```
# Run 1 iteration of SDXL-Turbo model
python main.py --model=sdxl-turbo

# Run 3 iterations of Resnet50 and Bert-Large on nc05 server
python main.py --server_id="nc05" --model=resnet50,bert-large --iterations=3

```

All model logs are outputted to `[SERVER_ID]_logs/` and power draw metrics to `[SERVER_ID]_monitor/` in the working directory.

## Citation

```bibtex
    @thesis{XiaJulia2025,
    author       = {JuliaXia},
    title        = {Schrödinger’s Carbon: Until Measured, Operational Emissions Remain Uncertain},
    school       = {MIT},
    year         = {2025},
    address      = {Cambridge},
    month        = {May},
    }
```

## References

[1] V. J. Reddi et al. MLPerf Inference Benchmark. 2019

[2] Luccioni, Gamazaychikov, Strubell, Hooker, Jernite, Mitchell, Wu. AI Energy Score. Hugging Face, 2024. url: https://huggingface.github.io/AIEnergyScore/

[3] NVIDIA Corporation. NVIDIA System Management Interface (nvidia-smi). url: https://developer.nvidia.com/system-management-interface.

[4] K. Keahey et al. “Lessons Learned from the Chameleon Testbed”. In: Proceedings of the 2020 USENIX Annual Technical Conference (USENIX ATC ’20). USENIX Association, July 2020
