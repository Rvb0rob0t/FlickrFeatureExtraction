import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from handlers import data_generator, model_builder
from utils import utils


# config
_BASE_MODEL_NAME = 'MobileNet'
_ROOT = os.path.dirname(__file__)
_AESTHETIC_WEIGHTS_FILE = os.path.join(
    _ROOT, 'models', _BASE_MODEL_NAME, 'weights_mobilenet_aesthetic_0.07.hdf5')
_TECHNICAL_WEIGHTS_FILE = os.path.join(
    _ROOT, 'models', _BASE_MODEL_NAME, 'weights_mobilenet_technical_0.11.hdf5')
_IMG_LOAD_DIMS = (224, 224)


def _image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


class NimaScorer:
    def __init__(self, tech=False, img_format="jpg"):
        # build model and load weights
        self.nima = model_builder.Nima(_BASE_MODEL_NAME, weights=None)
        self.nima.build()
        if tech:
            weights_filepath = _TECHNICAL_WEIGHTS_FILE
        else:
            weights_filepath = _AESTHETIC_WEIGHTS_FILE
        self.nima.nima_model.load_weights(weights_filepath)
        self.img_format = img_format

    def score(self, image_path):
        # image_dir, samples = _image_file_to_json(image_path)
        # test_data_generator = data_generator.TestDataGenerator(
        #     samples, image_dir, 1, 10, self.nima.preprocessing_function(), img_format=self.img_format)
        # predictions = self.nima.nima_model.predict_generator(
        #     test_data_generator, workers=1, use_multiprocessing=False, verbose=0)
        img_array = utils.load_image(image_path, _IMG_LOAD_DIMS)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = self.nima.preprocessing_function()(img_batch)
        pred = self.nima.nima_model(img_preprocessed)
        return utils.calc_mean_score(pred)


if __name__ == '__main__':
    nima_scorer = NimaScorer(tech=False)
    nima_tech_scorer = NimaScorer(tech=True)
    image_path = "/home/ruben/Photography_Capabilities_Case_Study/NIMA/idealo/image-quality-assessment/readme_figures/images_aesthetic/aesthetic2.jpg"
    score = nima_scorer.score(image_path)
    tech_score = nima_tech_scorer.score(image_path)
    print(f"Image scored with\naesthetic: {score}\ntechnical: {tech_score}")
