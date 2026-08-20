Uses [OpenCheetah](https://github.com/Alibaba-Gemini-Lab/OpenCheetah/tree/main),
[SecONNds](https://github.com/SecONNds/SecONNds_1_25/tree/main),
[Troy-Nova](https://github.com/lightbulb128/troy-nova/tree/master) and
[EMP-OT](https://github.com/emp-toolkit/emp-ot)
to implement protocols for 2PC.


# Overview

1. [Building](#building)
    1. [Dependencies](#build-the-dependencies)
    2. [This project](#build-the-project)
2. [Running](#run-the-project)
3. [Protocols](#protocols)
    1. [AB](#protocol-1)
    2. [AB2](#protocol-2)

# Building

## Build the dependencies

- Install OpenSSL and Eigen3
```sh
# for debian based distributions
apt install libssl-dev libeigen3-dev
```

- To install EMP-Tool/EMP-OT/SEAL use:
```sh
./deps.sh
```
$\Rightarrow$ this will create `./deps/` and place the projects inside it.

- For GPU support instead use the following to install [Troy-Nova](https://github.com/lightbulb128/troy-nova) (this requires CUDA)
    - (optional): Set [CUDA architecture](https://developer.nvidia.com/cuda-gpus) with `GPU_ARCHITECTURE`; default: `"75"`
```sh
GPU_ARCHITECTURE="<architecture>" ./deps.sh -gpu

# or try

GPU_ARCHITECTURE="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g')" ./deps.sh -gpu
```


## Build the project
```sh
./build.sh

# OR

mkdir ./data # to store ferret output
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_APPROX_RESHARE=OFF \
    -DTRIPLE_VERIFY=OFF -DTRIPLE_COLOR=OFF -DTRIPLE_ZERO=ON \
    -DTRIPLE_GPU=OFF
cmake --build build -j
```

CMake Options:
- `TRIPLE_VERIFY` (BOOL): If enabled - Verify correctness of the triples. (For testing only)
- `TRIPLE_COLOR` (BOOL): If enabled - Use ANSI color codes for colored logs.
- `TRIPLE_ZERO` (BOOL): If enabled - Allow tensors to be zero (can be insecure).
- `TRIPLE_GPU` (BOOL): If enabled - Use [Troy-Nova](https://github.com/lightbulb128/troy-nova) for Convolutions.
- `TRIPLE_GPU_REVERSE` (BOOL): If enabled - Encrypt filters instead of images (requires `TRIPLE_GPU=ON`).
- `TRIPLE_SEED` (NUM): Set the seed (-1: no seed).
- `TRIPLE_BITLEN` (NUM): Set the bitlength (default: 32).
- `TRIPLE_LOG_LEVEL` (NUM):
    - `0`: Log time and data required.
    - `1`: Log debug info and errors.
    - `2`: Log errors.

Formatting:
```sh
cmake --build build -t format
```


# Run the Project

Server:
```sh
./build/bin/cheetah <PORT> <SAMPLES> <BATCHSIZE> <threads>
```

Client:
```sh
./build/bin/cheetah_client <PORT> <HOST> <SAMPLES> <BATCHSIZE> <threads>
```

> [!NOTE]
> `<SAMPLES>` has currently no effect


# Protocols

Currently supported:
- Boolean triples via OT
- Matrix $\times$ Vector triples via HE
- 2D-Convolution triples via HE
- 2D-BatchNorm triples via HE

## Protocol 1

> [!NOTE]
> Both parties have shares of the image (A1/A2) and the filters (B1/B2)
> - Party1:
>   - IN: $A1, B1$
>   - OUT: $C1 = M2 + R1$
> - Party2:
>   - IN: $A2, B2$
>   - OUT: $C2 = M1 + R2$
>
> Verify: $C1 + C2 = (A1 + A2) \odot (B1 + B2)$

Step 1:
- Party 1: A1' = Enc(A1), send A1' to Party 2
- Party 2: A2' = Enc(A2), send A2' to Party 1

Step 2:
- Party 1: M1 = (A1 + A2') B1 - R1, send M1 to Party 2
- Party 2: M2 = (A1' + A2) B2 - R2, send M2 to Party 1

Step 2:
- Party 1: C1 = Dec(M2) + R1
- Party 2: C2 = DEC(M1) + R2

## Protocol 2

> [!NOTE]
> Both parties have shares of the image (A1/A2) but only `Party 2` has the filters (B2)
> - Party1:
>   - IN: $A1$
>   - OUT: $C1 = M2$
> - Party2:
>   - IN: $A2, B2$
>   - OUT: $C2 = R2$
>
> Verify $C1 + C2 = (A1 + A2) \odot B2$

Step 1:
- Party 1: A1' = Enc(A1), send A1' to Party 2

Step 2:
- Party 2: M2 = (A1' + A2) B2 - R2, send M2 to Party 1

Step 3:
- Party 1: C1 = Dec(M2) 
- Party 2: C2 = R2
