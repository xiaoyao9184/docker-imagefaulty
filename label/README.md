<!--
---
title: Classification images with ImageFaulty
type: guide
tier: all
order: 103
hide_menu: true
hide_frontmatter_title: true
meta_title: ImageFaulty model connection for classification images
meta_description: The ImageFaulty model connection integrates the capabilities of ImageFaulty with Label Studio to assist in machine learning labeling tasks involving ImageFaulty classification.
categories:
    - Computer Vision
    - ImageFaulty
image: "/tutorials/image-faulty.png"
---
-->

# ImageFaulty model connection

The [ImageFaulty](https://github.com/DALAI-project/FaultyImageAPI) model connection is a powerful tool that integrates the capabilities of ImageFaulty with Label Studio. It is designed to assist in machine learning labeling tasks, especially for tasks involving the classification of scanned document quality.

The primary function of this connection is to identify and scan empty documents, folded corners, handwritten and printed fonts, and label obstructions in images, which are crucial steps in many document scan quality inspection workflows. By automating this process, the ImageFaulty model connection can significantly increase efficiency, reducing the time and effort required for manual identification.

In the context of Label Studio, this connection enhances the platform's labeling capabilities, allowing users to automatically generate quality labels for images. This is particularly useful for tasks such as digital scanning.

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).

This tutorial uses the [`imagefaulty` example](https://github.com/xiaoyao9184/docker-imagefaulty/tree/main/label).

## Labeling configuration

The ImageFaulty model connection needs to be used with the labeling configuration in Label Studio. The configuration uses the following labels:

```xml
<View>
  <Image name="image" value="$image"/>

  <Header value="writing-type:"/>
  <Choices name="writing_type" toName="image" choice="single-radio">
    <Choice alias="H" value="Handwritten" />
    <Choice alias="T" value="Typewritten" />
    <Choice alias="C" value="Combination" />
  </Choices>
  <Header value="post-it:"/>
  <Choices name="post_it" toName="image" choice="single-radio">
    <Choice alias="NP" value="ok" />
    <Choice alias="P" value="post-it" />
  </Choices>
  <Header value="corner:"/>
  <Choices name="corner" toName="image" choice="single-radio">
    <Choice alias="NC" value="ok" />
    <Choice alias="C" value="folded_corner" />
  </Choices>
  <Header value="empty:"/>
  <Choices name="empty" toName="image" choice="single-radio">
    <Choice alias="NE" value="ok" />
    <Choice alias="E" value="empty" />
  </Choices>
</View>
```


> Warning! Please note that the current implementation of the ImageFaulty model connection does not support images that are directly uploaded to Label Studio. It is designed to work with images that are hosted publicly on the internet. Therefore, to use this connection, you should ensure that your images are publicly accessible via a URL.


## Running with Docker (recommended)

1. Start the Machine Learning backend on `http://localhost:9090` with the prebuilt image:

```bash
cd docker/up.label@gpu
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Create a project in Label Studio. Then from the **Model** page in the project settings, [connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio). The default URL is `http://localhost:9090`.


## Building from source (advanced)

To build the ML backend from source, you have to clone the repository and build the Docker image:

```bash
docker build -t xiaoyao9184/image-faulty:main -f ./docker/build@source/dockerfile .
```

## Running without Docker (advanced)

To run the ML backend without Docker, you have to clone the repository and install all dependencies using conda:

```bash
conda env create -f ./environment.yml
```

Then you can start the ML backend:

```bash
conda activate faulty_api_env
label-studio-ml start --root-dir . label
```

The ImageFaulty model connection offers several configuration options that can be set in the `docker-compose.yml` file:

- `BASIC_AUTH_USER`: Specifies the basic auth user for the model server.
- `BASIC_AUTH_PASS`: Specifies the basic auth password for the model server.
- `LOG_LEVEL`: Sets the log level for the model server.
- `WORKERS`: Specifies the number of workers for the model server.
- `THREADS`: Specifies the number of threads for the model server.
- `MODEL_DIR`: Specifies the model directory.
- `LABEL_STUDIO_ACCESS_TOKEN`: Specifies the Label Studio access token.
- `LABEL_STUDIO_HOST`: Specifies the Label Studio host.

These options allow you to customize the behavior of the ImageFaulty model connection to suit your specific needs.

# Customization

The ML backend can be customized by adding your own models and logic inside the `./label` directory. 