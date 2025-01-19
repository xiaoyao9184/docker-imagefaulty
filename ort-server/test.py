import os
import PIL
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

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

writing_type_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.882, 0.883, 0.899], [0.088, 0.089, 0.094])
    ])

# just for test
import pytest
import requests
import json
from urllib.parse import urljoin
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(str(pytestconfig.rootdir), "docker", "up.orts@cpu", "docker-compose.yml")

@pytest.fixture(scope="session")
def http_service(docker_ip, docker_services):

    port = docker_services.port_for("ort-server", 8080)
    url = "http://{}:{}".format(docker_ip, port)
    request_session = requests.Session()
    retries = Retry(total=50,
                    backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504])
    request_session.mount('http://', HTTPAdapter(max_retries=retries))

    assert request_session.get("{}/health".format(url))
    return request_session, url

@pytest.fixture
def image_path(pytestconfig):
    return os.path.join(str(pytestconfig.rootdir), "ort-server", "test_images")

def test_writing_type(http_service, image_path):
    request_session, api_url = http_service

    for idx, img in enumerate(["writing_type.0.jpg", "writing_type.1.jpg"]):
        image = PIL.Image.open(os.path.join(image_path, img))

        # Transformations for corner page detection
        input_image = data_transforms(image).unsqueeze(0)

        # name: input
        # tensor: float32[batch_size,3,224,224]
        request = {
            'input': input_image.tolist()
        }

        # name: output
        # tensor: float32[batch_size,2]
        item = request_session.post(urljoin(api_url, 'api/sessions/writing_type/v1'), data=json.dumps(request)).json()
        assert np.argmax(item['output'][0]) == idx

def test_post_it(http_service, image_path):
    request_session, api_url = http_service

    for idx, img in enumerate(["post_it.0.jpg"]):
        image = PIL.Image.open(os.path.join(image_path, img))

        # Transformations for corner page detection
        input_image = data_transforms(image).unsqueeze(0)

        # name: input
        # tensor: float32[batch_size,3,224,224]
        request = {
            'input': input_image.tolist()
        }

        # name: output
        # tensor: float32[batch_size,2]
        item = request_session.post(urljoin(api_url, 'api/sessions/post_it/v1'), data=json.dumps(request)).json()
        assert np.argmax(item['output'][0]) == idx

def test_corner(http_service, image_path):
    request_session, api_url = http_service

    for idx, img in enumerate(["corner.0.jpg", "corner.1.jpg"]):
        image = PIL.Image.open(os.path.join(image_path, img))

        # Transformations for corner page detection
        input_image = data_transforms(image).unsqueeze(0)

        # name: input
        # tensor: float32[batch_size,3,224,224]
        request = {
            'input': input_image.tolist()
        }

        # name: output
        # tensor: float32[batch_size,2]
        item = request_session.post(urljoin(api_url, 'api/sessions/corner/v1'), data=json.dumps(request)).json()
        assert np.argmax(item['output'][0]) == idx

def test_empty(http_service, image_path):
    request_session, api_url = http_service

    for idx, img in enumerate(["empty.0.jpg", "empty.1.jpg"]):
        image = PIL.Image.open(os.path.join(image_path, img))

        # Transformations for corner page detection
        input_image = data_transforms(image).unsqueeze(0)

        # name: input
        # tensor: float32[batch_size,3,224,224]
        request = {
            'input': input_image.tolist()
        }

        # name: output
        # tensor: float32[batch_size,2]
        item = request_session.post(urljoin(api_url, 'api/sessions/empty/v5_24_08_23'), data=json.dumps(request)).json()
        assert np.argmax(item['output'][0]) == idx
