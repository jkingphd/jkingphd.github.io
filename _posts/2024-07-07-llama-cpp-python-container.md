---
layout: post
title:  "Building a llama-cpp-python Docker image with GPU support"
date:   2024-07-07 21:45:40 -0500
tags: llama.cpp LLM
---

## Overview

For my first post, I thought I would dive right into an ongoing project. Without getting too into the weeds, I submitted a talk to the Nashville Innovation Summit on Labeling with Large Language Models (LLMs). This closely aligns with some of the work we have been doing at Xsolis, namely using LLMs to extract labels that would otherwise be too difficult and/or time-consuming to extract by hand. A big component of that project is experimenting with locally-hosted LLMs, so I wanted to experiment with some different methods in my home lab.

## Plan

The general idea was the try and get llama-cpp-python working with GPU support so I could start experimenting with some labeling optimizations (e.g., batching, pre-caching, grammar, etc.). 

1. Pull an nvidia/cuda image and attempt to install llama-cpp-python on it.
2. Pull a llama.cpp image and attempt to install llama-cpp-python on it without building llama.cpp again.
3. Build an image from scratch.

Spoiler, I landed on Option 3. If you just want the code, feel free to skip ahead.

## Results

# Pre-built nvidia/cuda image

This was a mess of dependencies. First, I tried to pull the latest version fo the nvidia/cuda Ubuntu image `nvidia/cuda:12.5.0-devel-ubuntu22.04`, but on launch I got the following error:

```bash
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.5, please update your driver to a newer version, or use an earlier cuda container: unknown.
```

This is likely an issue with my late-model 1080 Ti, and I wasted a good deal of time trying to update my nvidia drivers to nvidia-driver-545 (which isn't even the newest version, I'll add). After that mess, I reverted to nvidia-driver-535 and started to use an older version of the nvidia/cude image from last year before deciding that this way lay madness. On to Option 2!

# Pre-built llama.cpp image

I've had a lot of success running the built-in [HTTP server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md) included with llama.cpp. Specifically, the pre-built [Docker images](https://github.com/ggerganov/llama.cpp/blob/master/docs/docker.md) for llama.cpp work very well across a variety of different GPU configurations I have used (e.g., 1080 Ti, 2080 Ti, Titan Xp, A5000, A6000, T4, A10G, etc.). If you simply need an OpenAI-compatable API, I would recommend leveraging `ghcr.io/ggerganov/llama.cpp:server-cuda`. However, I'm interested in embedding llama.cpp into my application, so let's try and piggyback off of Gerganov's excellent work and install llama-cpp-python without rebuilding llama.cpp from scratch.

```dockerfile
# Grabbing the image with full CUDA support, in case we need development libraries not in the lighter images.
FROM ghcr.io/ggerganov/llama.cpp:full-cuda

ENV LLAMA_CPP_LIB=/app/libllama.so

RUN apt update && \
    apt install -y python-is-python3 && \
    # Basing this code off of a [comment](https://github.com/abetlen/llama-cpp-python/issues/1070#issuecomment-1881737418) by [abetlen](https://github.com/abetlen) on the llama-cpp-python issues page.
    make BUILD_SHARED_LIBS=1 GGML_CUDA=1 libllama.so -j && \
    CMAKE_ARGS="-DLLAMA_BUILD=OFF" pip install llama-cpp-python --no-cache-dir
```

That seems to run fine, but now when I start a container and attempt to run inference, I get the following:

```python
>>> from llama_cpp import Llama
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/llama_cpp/llama_cpp.py", line 75, in _load_shared_library
    return ctypes.CDLL(str(_lib_path), **cdll_args)  # type: ignore
  File "/usr/lib/python3.10/ctypes/__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libggml.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.10/dist-packages/llama_cpp/__init__.py", line 1, in <module>
    from .llama_cpp import *
  File "/usr/local/lib/python3.10/dist-packages/llama_cpp/llama_cpp.py", line 88, in <module>
    _lib = _load_shared_library(_lib_base_name)
  File "/usr/local/lib/python3.10/dist-packages/llama_cpp/llama_cpp.py", line 77, in _load_shared_library
    raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")
RuntimeError: Failed to load shared library '/app/libllama.so': libggml.so: cannot open shared object file: No such file or directory
```

I spent a while troubleshooting this, changing the LLAMA_CPP_LIB environmental variable to libggml.so and attempting to make the libraries with different flags to no avail. The answer seemed close, but after an hour or so, I thought it would be best to try and build the image from scratch, since the llama-cpp-python install seems to be pretty robust, based on my memory of installing it on bare metal. 

# Build an image from scratch

Let's just bite the bullet and build the danged thing from scratch.

```dockerfile
FROM ubuntu:22.04

## Install CUDA Toolkit (Includes drivers and SDK needed for building llama-cpp-python with CUDA support)
RUN apt update && \
    apt install -y python3 python3-pip python-is-python3 gcc wget git && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda-repo-ubuntu2204-12-5-local_12.5.1-555.42.06-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-5-local_12.5.1-555.42.06-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt update && \
    apt install -y cuda-toolkit-12-5

## Install llama-cpp-python with CUDA Support (and jupyterlab)
RUN CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all-major" \
    FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
```

I was then able to verify that I could load Mistral into GPU and generate a response:

```bash
docker run -it --rm --gpus all -v ~/models:/models/ llama-cpp-python-cuda
python
```

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    n_gpu_layers=-1
)

output = llm("How many planets are in our solar system?")
print(output['choices'][0]['text'])
```

The output is limited to 16 tokens by defaults, but the results look decent : `Nine. The eight inner planets are Mercury, Venus, Earth,`. More importantly, I am seeing that the GPU is being used within nvtop:

```
 Device 0 [NVIDIA GeForce GTX 1080 Ti] PCIe GEN 3@16x RX: 0.000 KiB/s TX: 0.000 KiB/s
 GPU 1480MHz MEM 5005MHz TEMP  38°C FAN  23% POW  56 / 250 W
 GPU[                                  0%] MEM[||||||||||||||       4.397Gi/11.000Gi]
   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
100│GPU0 %                                                                                                                        │
   │GPU0 mem%                                                                                                                     │
   │                                                                                                                              │
   │                                                                                                                              │
   │                                                                                                                              │
 75│                                                                                                                              │
   │                                                                                                                              │
   │                                                                                                                              │
   │                                                                                                                              │
   │                                                                                                  ┌─┐                         │
 50│                                                                                                  │ │                         │
   │                                                                                                  │ │                         │
   │                                                                 ┌────────────────────────────────┼─┼─────────────────────────│
   │                                                                 │                                │ │                         │
   │                                                                 │                                │ │                         │
 25│                                                                 │                                │ │                         │
   │                                                                 │                                │ │                         │
   │                                                                 │                                │ │                         │
   │                                                                 │                                │ │                         │
   │                                                                 │                                │ │                         │
  0│─────────────────────────────────────────────────────────────────┴────────────────────────────────┘ └─────────────────────────│
   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    PID USER DEV    TYPE  GPU        GPU MEM    CPU  HOST MEM Command
   3493 root   0 Compute   0%   4406MiB  39%     1%    396MiB python
```

Some cleanup items for the future, in order of priority:
- There are a ton of artifacts in the above image, including a bunch of install files I really don't need in a base image. I probably need to leverage some --no-cache options and/or consider a multi-stage build.
- I don't like the idea of having to manually change the CUDA install file when the version increments. I may look for some mechanism to semi-automate that step in the build.

## Conclusion

In all, this was simply step zero for my project, so I don't want to spend a great deal of time worrying about optimizing the build. I will call this a success and move onto the actual labeling work now.