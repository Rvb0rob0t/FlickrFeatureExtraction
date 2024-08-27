# Flickr Feature Extraction

A robust toolkit designed for extracting, processing, and scoring features from Flickr photos using its API. This project integrates advanced image processing models to evaluate aesthetic qualities through AI, providing a customizable solution for researchers and developers.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Caffe](https://caffe.berkeleyvision.org/)

### Installation

Follow these steps to set up the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/flickr-feature-extraction.git
   cd flickr-feature-extraction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**:
   - Obtain your Flickr API key and secret from [Flickr's API portal](https://www.flickr.com/services/api/).
   - Create a `.env` file in the root directory with your keys:
     ```
     FLICKR_KEY=your_flickr_key
     FLICKR_SECRET=your_flickr_secret
     ```

4. **Configure the project**:
   - Edit the `config.ini` file to set paths, image formats, and other parameters to suit your needs.

5. **Ready to extract features from Flickr**:
   ```bash
   python multithreaded.py
   ```

## Key Features

- **Flickr API Integration**: Retrieve and process photos and user data directly from Flickr with ease.
- **Advanced Image Processing**: Assess images using cutting-edge models like NIMA and ShuKong Aesthetic Scorer for detailed aesthetic evaluations.
- **Robust Error Handling**: Ensures continuous operation even when encountering API limitations or network issues.
- **Extensive Feature Extraction**: Extract a wide range of features, including EXIF data, user details, and image metadata.
- **Configurable**: Easily customize the toolkit through a simple configuration file.

## Usage

### Basic Feature Extraction

To extract features from a user's photos:

```python
from flickr_feature_extraction import FlickrFeatureExtraction

config_path = 'path_to_your_config.ini'
env_path = '.env'

extractor = FlickrFeatureExtraction(config_path, env_path)
extractor.full_persist_user_and_photo_sample_features(user_id='example_user_id')
```

### Authentication

Authenticate with Flickr via the browser:

```python
extractor.authenticate_via_browser(perms='read')
```

### Image Scoring

Automatically score images using pre-trained models:

- **NIMA**: Evaluates overall aesthetic quality.
- **ShuKong**: Focuses on specific aesthetic attributes.

## Authors and affiliations

This project is a collaborative effort by:

- Rubén Gaspar Marco
- Sofia Strukova
- José A. Ruipérez-Valiente
- Félix Gómez Mármol

The authors are affiliated with the Department of Information and Communications Engineering, University of Murcia, Spain.
