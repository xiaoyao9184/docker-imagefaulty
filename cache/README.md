# cache

This folder is the cache directory for Hugging Face (HF).

When using online mode, downloaded models will be cached in this folder.

For [offline mode](https://huggingface.co/docs/transformers/main/installation#offline-mode) use, please download the models in advance and specify the model directory,
such as the `xiaoyao9184/image-faulty` model below.

The folder structure for `./cache/huggingface/hub/models--xiaoyao9184--image-faulty` is as follows.

```
.
├── blobs
│   ├── 05ceaff0dd7a51ee0e94c932bbb2d16505dc9e744bff93c7d4da1b5498056032
│   ├── 7a2df50881906e67b5f5f3985a0ac20f288de4064163362fffa96ec01ecbc796
│   ├── 9f8f9aa5623e159f4f4be9a24453132112cc4a05
│   ├── a118d0ca8c0e58e57c3a81d3b2e8e6a605080dc7
│   ├── a6344aac8c09253b3b630fb776ae94478aa0275b
│   ├── ec107f0c971df4745fc5b47256d06e6f45355e0f24f5245e92cd4e7eaba4ca53
│   └── f43fa4d3390a4efe97dd54529d96e2eaea3b841cb01abf1b6672a35eb6ffeffe
├── refs
│   └── main
└── snapshots
    └── 3bc9ba7f84a5d8359a34ef3a7d5e9f6bec349d91
        ├── corner_model.onnx -> ../../blobs/ec107f0c971df4745fc5b47256d06e6f45355e0f24f5245e92cd4e7eaba4ca53
        ├── empty_v5_24_08_23.onnx -> ../../blobs/7a2df50881906e67b5f5f3985a0ac20f288de4064163362fffa96ec01ecbc796
        ├── info.txt -> ../../blobs/9f8f9aa5623e159f4f4be9a24453132112cc4a05
        ├── post_it_model.onnx -> ../../blobs/05ceaff0dd7a51ee0e94c932bbb2d16505dc9e744bff93c7d4da1b5498056032
        ├── README.md -> ../../blobs/a118d0ca8c0e58e57c3a81d3b2e8e6a605080dc7
        └── writing_type_v1.onnx -> ../../blobs/f43fa4d3390a4efe97dd54529d96e2eaea3b841cb01abf1b6672a35eb6ffeffe

5 directories, 14 files
```

It will use
- `./cache/huggingface/hub/models--xiaoyao9184--image-faulty/snapshots/3bc9ba7f84a5d8359a34ef3a7d5e9f6bec349d91`

For more details, refer to [up@cpu/docker-compose.yml](./../docker/up@cpu/docker-compose.yml).


## Pre-download for offline mode

Running in online mode will automatically download the model.

install cli

```bash
pip install -U "huggingface_hub[cli]"
```

download model

```bash
huggingface-cli download xiaoyao9184/image-faulty --repo-type model --revision main --cache-dir ./cache/huggingface/hub
```