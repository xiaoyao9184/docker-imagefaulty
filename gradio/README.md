---
title: Image Faulty
emoji: 📚
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.28.0
app_file: gradio_app.py
pinned: false
license: apache-2.0
short_description: Gradio implementation of the FaultyImageAPI
models:
  - xiaoyao9184/image-faulty
preload_from_hub:
  - xiaoyao9184/image-faulty
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Need set `MODEL_PATH` to point to the path of a snapshot of the model you want to use, like `/home/user/.cache/huggingface/hub/models--xiaoyao9184--image-faulty/snapshots/3bc9ba7f84a5d8359a34ef3a7d5e9f6bec349d91`.


## MCP prompt usage

server setup see [mcp-server-with-gradio](https://www.gradio.app/guides/building-mcp-server-with-gradio)

```prompt
Detect the image of corner https://raw.githubusercontent.com/xiaoyao9184/docker-imagefaulty/refs/heads/main/ort-server/test_images/corner.1.jpg
```

```prompt
Detect the image of writing type, post-it notes, corner, and empty https://raw.githubusercontent.com/xiaoyao9184/docker-imagefaulty/refs/heads/main/ort-server/test_images/corner.1.jpg
```
