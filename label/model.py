
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import os
import onnxruntime
import numpy as np
import logging

from PIL import Image
from uuid import uuid4
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

logger = logging.getLogger(__name__)


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

class ImageFaulty(LabelStudioMLBase):
    """Custom ML Backend model
    """
    # Label Studio image upload folder:
    # should be used only in case you use direct file upload into Label Studio instead of URLs
    LABEL_STUDIO_ACCESS_TOKEN = (
        os.environ.get("LABEL_STUDIO_ACCESS_TOKEN") or os.environ.get("LABEL_STUDIO_API_KEY")
    )
    LABEL_STUDIO_HOST = (
        os.environ.get("LABEL_STUDIO_HOST") or os.environ.get("LABEL_STUDIO_URL")
    )

    MODEL_DIR = os.getenv('MODEL_DIR', '.')

    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def predict_single(self, task):
        logger.debug('Task data: %s', task['data'])

        labels_w = []
        labels_p = []
        labels_c = []
        labels_e = []
        for l in self.label_interface.labels:
            for key, value in l.items():
                if value.parent_name == 'writing_type':
                    labels_w.append(value.value)
                elif value.parent_name == 'post_it':
                    labels_p.append(value.value)
                elif value.parent_name == 'corner':
                    labels_c.append(value.value)
                elif value.parent_name == 'empty':
                    labels_e.append(value.value)
        if len(labels_w) == 0 \
            and len(labels_p) == 0 \
            and len(labels_c) == 0 \
            and len(labels_e) == 0:
            logger.error("No labels found. Please check your label configuration. You should have at least one 'writing type', 'post-it', 'corner', 'empty' label in the tag.")

        from_name_w, to_name_w, value_w = self.get_first_tag_occurence('Choices', 'Image', name_filter=lambda x: x == 'writing_type')
        from_name_p, to_name_p, value_p = self.get_first_tag_occurence('Choices', 'Image', name_filter=lambda x: x == 'post_it')
        from_name_c, to_name_c, value_c = self.get_first_tag_occurence('Choices', 'Image', name_filter=lambda x: x == 'corner')
        from_name_e, to_name_e, value_e = self.get_first_tag_occurence('Choices', 'Image', name_filter=lambda x: x == 'empty')

        enable_w = from_name_w is not None
        enable_p = from_name_p is not None
        enable_c = from_name_c is not None
        enable_e = from_name_e is not None

        value = value_w if enable_w else value_p if enable_p else value_c if enable_c else value_e

        id_gen = str(uuid4())[:4]

        image_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)
        image_path = get_local_path(image_url, task_id=task.get('id'))

        # run detect
        img_pil = Image.open(image_path).convert("RGB")
        res_dict = detect(img_pil, enable_w, enable_p, enable_c, enable_e)

        result = []
        if enable_w:
            result.append({
                'id': id_gen,
                'from_name': from_name_w,
                'to_name': to_name_w,
                'value': {
                    'choices': [
                        labels_w[res_dict['writing_type']]
                    ]
                },
                'type': 'choices'
            })
        if enable_p:
            result.append({
                'id': id_gen,
                'from_name': from_name_p,
                'to_name': to_name_p,
                'value': {
                    'choices': [
                        labels_p[res_dict['post_it']]
                    ]
                },
                'type': 'choices'
            })
        if enable_c:
            result.append({
                'id': id_gen,
                'from_name': from_name_c,
                'to_name': to_name_c,
                'value': {
                    'choices': [
                        labels_c[res_dict['corner']]
                    ]
                },
                'type': 'choices'
            })
        if enable_e:
            result.append({
                'id': id_gen,
                'from_name': from_name_e,
                'to_name': to_name_e,
                'value': {
                    'choices': [
                        labels_e[res_dict['empty']]
                    ]
                },
                'type': 'choices'
            })

        return {
            'result': result,
            'score': 1,
            'model_version': self.get('model_version'),
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        for task in tasks:
            # TODO: implement is_skipped() function
            # if is_skipped(task):
            #     continue

            prediction = self.predict_single(task)
            if prediction:
                predictions.append(prediction)

        return ModelResponse(predictions=predictions, model_versions=self.get('model_version'))
