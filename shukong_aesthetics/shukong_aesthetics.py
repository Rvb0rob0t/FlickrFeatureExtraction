import os

import numpy as np

import caffe
import cv2
from caffe.proto import caffe_pb2

# config
ROOT = os.path.dirname(__file__)
ARCHITECTURE = os.path.join(ROOT, 'model', 'initModel.prototxt')
WEIGHTS = os.path.join(ROOT, 'model', 'initModel.caffemodel')
IMAGE_MEAN = os.path.join(ROOT, 'model', 'mean_AADB_regression_warp256.binaryproto')

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

INPUT_LAYER = 'imgLow'

def transform_img(img, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    '''Image processing helper function'''
    #Image Resizing
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
    return img

class ShuKongAestheticScorer:
    """
    Shu Kong aesthetic rate predictor
    """
    def __init__(self):
        self.net = caffe.Net(ARCHITECTURE, WEIGHTS, caffe.TEST)
        mean_blob = caffe_pb2.BlobProto()
        with open(IMAGE_MEAN, 'rb') as f:
            mean_blob.ParseFromString(f.read())
        mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
            (mean_blob.channels, mean_blob.height, mean_blob.width))
        mean_array = mean_array[:, 15:242,15:242]

        # Define image transformers
        self.net.blobs[INPUT_LAYER].reshape(
            1,  # batch size
            3,  # channel
            IMAGE_WIDTH, IMAGE_HEIGHT)  # image size
        self.transformer = caffe.io.Transformer(
            {INPUT_LAYER: self.net.blobs[INPUT_LAYER].data.shape})
        self.transformer.set_mean(INPUT_LAYER, mean_array)
        self.transformer.set_transpose(INPUT_LAYER, (2,0,1))

    def score(self, pil_img):
        """Scores image in PIL format.
        """
        img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        img = transform_img(img)

        self.net.blobs[INPUT_LAYER].data[...] = self.transformer.preprocess(INPUT_LAYER, img)
        out = self.net.forward()
        return out['fc11_score'][0][0]

if __name__ == "__main__":
    from PIL import Image

    kongaesthetics = ShuKongAestheticScorer()
    p = "/home/ruben/Photography_Capabilities_Case_Study/NIMA/idealo/image-quality-assessment/readme_figures/images_aesthetic/aesthetic1.jpg"
    with Image.open(p) as img:
        score = kongaesthetics.score(img)
    print(f"Image scored with {score}")
