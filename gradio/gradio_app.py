
import gradio as gr

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import os
import onnxruntime
import numpy as np

def predict_fault(image, model):
    image = image.detach().cpu().numpy()
    input = {model.get_inputs()[0].name: image}
    output = model.run(None, input)
    preds = np.argmax(output[0], 1)
    return preds.item()

def detect(image, writing_type, post_it, corner, empty):
    writing_type_model, post_it_model, corner_model, empty_model = models
    
    res_dict = {}

    if writing_type:
        input_image = writing_type_transforms(image).unsqueeze(0)
        label = predict_fault(input_image, writing_type_model)
        res_dict['writing_type'] = label

    if post_it:
        input_image = data_transforms(image).unsqueeze(0)
        label = predict_fault(input_image, post_it_model)
        res_dict['post_it'] = label

    if corner:
        input_image = data_transforms(image).unsqueeze(0)
        label = predict_fault(input_image, corner_model)
        res_dict['corner'] = label

    if empty:
        input_image = empty_transforms(image).unsqueeze(0)
        label = predict_fault(input_image, empty_model)
        res_dict['empty'] = 1 - label

    return res_dict

def load_models():
    try:
        MODEL_PATH = os.environ.get("MODEL_PATH", './models/')
        POST_IT_MODEL = os.environ.get("POST_IT_MODEL", 'post_it_model.onnx')
        CORNER_MODEL = os.environ.get("CORNER_MODEL", 'corner_model.onnx')
        EMPTY_MODEL = os.environ.get("EMPTY_MODEL", 'empty_v5_24_08_23.onnx')
        WRITING_TYPE_MODEL = os.environ.get("WRITING_TYPE_MODEL", 'writing_type_v1.onnx')

        print(f"ORT device: {onnxruntime.get_device()}")

        # Load the models and the trained weights
        writing_type_model = onnxruntime.InferenceSession(os.path.join(MODEL_PATH, WRITING_TYPE_MODEL))
        post_it_model = onnxruntime.InferenceSession(os.path.join(MODEL_PATH, POST_IT_MODEL))
        corner_model = onnxruntime.InferenceSession(os.path.join(MODEL_PATH, CORNER_MODEL))
        empty_model = onnxruntime.InferenceSession(os.path.join(MODEL_PATH, EMPTY_MODEL))

        return writing_type_model, post_it_model, corner_model, empty_model
    except Exception as e:
        print("Failed to load pretrained models: {}".format(e))

# Load the models
models = load_models()

# Transform methods for corner & post-it model inputs
data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

# Transform methods for empty model inputs
empty_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Transform methods for writing-type model inputs
writing_type_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.882, 0.883, 0.899], [0.088, 0.089, 0.094])
    ])


with gr.Blocks(title="Image Faulty Demo") as demo:
    gr.Markdown("""
    # Image Faulty
    Find the project [here](https://github.com/xiaoyao9184/image-faulty).
    """)

    with gr.Row():
        with gr.Column():
            detecting_img = gr.Image(label="Input Image", type="pil", height=512)
        with gr.Column():
            writing_ckb = gr.Checkbox(label="Writing type", value=True)
            postit_ckb = gr.Checkbox(label="Post it", value=True)
            corner_ckb = gr.Checkbox(label="Folded corner", value=True)
            empty_ckb = gr.Checkbox(label="Parper empty", value=True)
            detecting_btn = gr.Button("Detect")
            predicted_messages = gr.JSON(label="Detected Messages")

    detecting_btn.click(
        fn=detect,
        inputs=[detecting_img, writing_ckb, postit_ckb, corner_ckb, empty_ckb],
        outputs=[predicted_messages]
    )

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
