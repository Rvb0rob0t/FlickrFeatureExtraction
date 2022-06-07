# Flickr Feature Extraction
Toolkit for extracting features from Flickr through its API.

## Getting Started

### Prerequisites
* tensorflow
* keras
* caffe

### Installation
```
$ Install prerequisites
$ Ready to extract features from Flickr
```

## Usage
```
$ nohup python multithreaded.py > /dev/null 2> stderr.log &
$ nohup python flickr_feature_extraction/assembly.py "final_features_14-02-2022/features/*/photo_features/*.json" -o final_features_14-02-2022/photo_features/ -s -3 > /dev/null 2> stderr.log &
```

## Additional Documentation and Acknowledgments
TODO

## Authors
- Rubén Gaspar Marco
- Sofia Strukova
- José A. Ruipérez-Valiente
- Félix Gómez Mármol

## Affiliations
Authors are with the Department of Information and Communications Engineering, University of Murcia, Spain
