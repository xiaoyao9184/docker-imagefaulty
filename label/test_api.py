"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""
import os.path

import pytest
import json
from model import ImageFaulty
import responses


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=ImageFaulty)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def model_dir_env(tmp_path, monkeypatch):
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    monkeypatch.setattr(ImageFaulty, 'MODEL_DIR', str(model_dir))
    return model_dir


@responses.activate
def test_predict(client, model_dir_env):
    responses.add(
        responses.GET,
        'http://test_predict.wam.ml-backend.com/corner.1.jpg',
        body=open(os.path.join(os.path.dirname(__file__), 'test_images', 'corner.1.jpg'), 'rb').read(),
        status=200
    )
    request = {
        'tasks': [{
            'data': {
                'image': 'http://test_predict.wam.ml-backend.com/corner.1.jpg'
            }
        }],
        # Your labeling configuration here
        'label_config': '''
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
'''
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    expected_texts = {
        'H',
        'NP',
        'C',
        'NE'
    }
    texts_response = set()
    for r in response['results'][0]['result']:
        if r['from_name'] == 'writing_type':
            assert r['value']['choices'] == ['H']
            texts_response.add(r['value']['choices'][0])
        elif r['from_name'] == 'post_it':
            assert r['value']['choices'] == ['NP']
            texts_response.add(r['value']['choices'][0])
        elif r['from_name'] == 'corner':
            assert r['value']['choices'] == ['C']
            texts_response.add(r['value']['choices'][0])
        elif r['from_name'] == 'empty':
            assert r['value']['choices'] == ['NE']
            texts_response.add(r['value']['choices'][0])
    assert texts_response == expected_texts
