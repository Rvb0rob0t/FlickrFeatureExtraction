import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image

from handlers import model_builder
from utils import utils


# config
_BASE_MODEL_NAME = 'MobileNet'
_ROOT = os.path.dirname(__file__)
_AESTHETIC_WEIGHTS_FILE = os.path.join(
    _ROOT, 'models', _BASE_MODEL_NAME, 'weights_mobilenet_aesthetic_0.07.hdf5')
_TECHNICAL_WEIGHTS_FILE = os.path.join(
    _ROOT, 'models', _BASE_MODEL_NAME, 'weights_mobilenet_technical_0.11.hdf5')
_IMG_LOAD_DIMS = (224, 224)
_PIL_INTERPOLATION_METHOD = Image.NEAREST


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

    def preprocess(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != _IMG_LOAD_DIMS:
            img = img.resize(_IMG_LOAD_DIMS, _PIL_INTERPOLATION_METHOD)
        img_array =  np.asarray(img)
        img_batch = np.expand_dims(img_array, axis=0)
        return self.nima.preprocessing_function()(img_batch)

    def score(self, img):
        """Scores image in PIL format.
        """
        img_preprocessed = self.preprocess(img)
        pred = self.nima.nima_model(img_preprocessed)
        return utils.calc_mean_score(pred)


if __name__ == '__main__':
    nima_scorer = NimaScorer(tech=False)
    nima_tech_scorer = NimaScorer(tech=True)
    image_path = "/home/ruben/Photography_Capabilities_Case_Study/NIMA/idealo/image-quality-assessment/readme_figures/images_aesthetic/aesthetic2.jpg"
    with Image.open(image_path) as img:
        score = nima_scorer.score(img)
        tech_score = nima_tech_scorer.score(img)
    print(f"Image scored with\naesthetic: {score}\ntechnical: {tech_score}")
