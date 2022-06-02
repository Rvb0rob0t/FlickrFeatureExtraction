# Adaptation from https://github.com/jeffheaton/pyimgdata
# Special thanks to: Jeff Heaton (http://www.heatonresearch.com)

import argparse
import configparser
import csv
import json
import logging
import logging.config
import logging.handlers
import math
import os
import random
import time
import webbrowser
from io import BytesIO

import dotenv
import flickrapi
import requests
from PIL import Image

import tracking
from nima import nima
from shukong_aesthetics import shukong_aesthetics

_CONTENT_TYPE = 1
_MEDIA_TYPE = 'photos'
_SIZE_TYPES = ['n', 'w', 'z', 'c', 'l', 'o']

_MAX_CONTACTS_PER_PAGE = 1000
_MAX_FAVORITES_PER_PAGE = 50

# Please note that Flickr will return at most the first 4,000 results
# for any given search query. If this is an issue, we recommend trying a more
# specific query. https://www.flickr.com/services/api/flickr.photos.search.htm
_MAX_N_RESULTS = 4000


def _sample_or_get_all(population, n):
    if len(population) <= n:
        return list(population)
    else:
        return random.sample(population, n)


def _count_walker_queries(result_length, per_page):
    return 1 if result_length == 0 else math.ceil(result_length/per_page)


class FlickrFeatureExtraction:

    session = requests.Session()

    def __init__(self, config_filepath, key_environment_filepath=None,
                 photo_scorers=None, tracker=None):
        self.logger = logging.getLogger(__name__)

        config_parser = configparser.ConfigParser()
        config_parser.read(config_filepath)

        io_config = config_parser['IO']
        self.input_path = io_config['input_path']
        self.output_path = io_config['output_path']
        self.image_format = io_config['image_format']

        requirements_config = config_parser['Requirements']
        self.licenses_allowed = requirements_config['licenses_allowed'].split(
            ',')
        self.photo_min_size = int(requirements_config['min_size'])
        self.photo_sample_size = int(requirements_config['photo_sample_size'])

        control_config = config_parser['Control']
        self.mins_to_update = int(control_config['minutes_to_update'])
        self.timeout = float(control_config['timeout'])

        secrets = dotenv.dotenv_values(key_environment_filepath)
        self.flickr = flickrapi.FlickrAPI(
            secrets['FLICKR_KEY'],
            secrets['FLICKR_SECRET'],
            timeout=self.timeout)

        if photo_scorers is not None:
            self.photo_scorers = photo_scorers
        else:
            self.photo_scorers = {}
            self.photo_scorers['kong'] = shukong_aesthetics.ShuKongAestheticScorer(
            )
            self.photo_scorers['nima'] = nima.NimaScorer(tech=False)
            self.photo_scorers['nima_tech'] = nima.NimaScorer(tech=True)

        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = tracking.Tracker()
        self.tracker.register_speeds(
            'photos_registered', 'queries')

    def authenticate_via_browser(self, perms='read'):
        self.logger.info("Starting authentication...")
        # Only do this if we don't have a valid token already
        if not self.flickr.token_valid(perms=perms):
            # Get a request token
            self.flickr.get_request_token(oauth_callback='oob')
            # Open a browser at the authentication URL.
            authorize_url = self.flickr.auth_url(perms=perms)
            webbrowser.open_new_tab(authorize_url)
            # Get the verifier code from the user.
            code = input("Verifier code: ")
            # Trade the request token for an access token
            self.flickr.get_access_token(code)

    def _insistent_call(self, requester, *args, **kwargs):
        while True:
            try:
                result = requester(*args, **kwargs)
                break
            except flickrapi.exceptions.FlickrError as e:
                if e.code is not None:
                    raise
                else:
                    self.logger.error(
                        ("Unexpected and unknown exception "
                         "probably flickr's fault. Trying again..."))
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout):
                self.logger.error(
                    f"Connection problems calling {requester.__name__} method. Trying again...",
                    exc_info=True)
        return result

    def _select_size(self, photo):
        selected_size = None
        for t in _SIZE_TYPES:
            url = photo.get('url_' + t)
            if url is None:
                self.logger.debug(
                    f"Size {t} not available for id={photo.get('id')}")
            else:
                selected_size = t
                height = int(photo.get('height_' + t))
                width = int(photo.get('width_' + t))
                if (width < self.photo_min_size) or (height < self.photo_min_size):
                    self.logger.debug(
                        f"Size {t}={width}x{height} too small for photo {photo.get('id')}")
                else:
                    break

        if selected_size is None:
            self.logger.warning(
                f"No size type is available and allowed for photo {photo.get('id')}")
        else:
            self.logger.debug(
                f"Size {t}={width}x{height} selected for photo {photo.get('id')}")
        return selected_size

    def _load_image(self, url, timeout=None, insistent=False):
        while True:
            try:
                response = self.session.get(
                    url, timeout=timeout or self.timeout)
                # Raises a HTTPError if the status is 4xx, 5xx
                response.raise_for_status()
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout):
                if insistent:
                    self.logger.error(
                        f"Connection problems downloading {url}. Trying again...",
                        exc_info=True)
                else:
                    raise
            except requests.exceptions.HTTPError:
                if (response.status_code // 100 == 5) and insistent:
                    self.logger.error("Server error. Trying again...")
                else:
                    raise
            else:
                img = Image.open(BytesIO(response.content))
                img.load()
                return img

    def download_photo(self, photo, insistent=True):
        license = photo.get('license')
        size_type = self._select_size(photo)

        image = None
        url = None
        if size_type is not None and license in self.licenses_allowed:
            url = photo.get('url_' + size_type)
            try:
                image = self._load_image(url, insistent=insistent)
            except:
                raise
            else:
                self.tracker.increment('photos_downloaded')
        else:
            self.tracker.increment('photos_rejected')

        return url, image

    def process_image(self, image):
        # Convert to full color (no grayscale, no transparent)
        if image.mode not in ('RGB'):
            self.logger.debug(f"Not RGB converted to RGB")
            rgbimg = Image.new('RGB', image.size)
            rgbimg.paste(image)
            image = rgbimg

        return image

    def get_basic_photo_features(self, photo_id):
        self.logger.debug(
            f"Starting to get flickr features for photo {photo_id}...")
        # t0 = time.perf_counter()
        features = {}
        photo = self.flickr.photos.getInfo(photo_id=photo_id)[0]
        self.tracker.increment('queries')

        features['id'] = photo.get('id')
        features['owner'] = photo.find('owner').get('nsid')
        features['title'] = photo.find('title').text
        features['description'] = photo.find('description').text
        features['views'] = photo.get('views')
        features['dateuploaded'] = photo.get('dateuploaded')
        features['lastupdate'] = photo.find('dates').get('lastupdate')
        # t1 = time.perf_counter()
        # self.logger.debug(f"getInfo: {t1-t0}")
        # t0 = t1

        features['tags'] = []
        for tag in photo.iter('tag'):
            # Lowercased and letters only from attrib `raw`
            features['tags'].append(tag.text)
        # t1 = time.perf_counter()
        # self.logger.debug(f"tags: {t1-t0}")
        # t0 = t1

        self.logger.debug(
            f"About to get list of comments of photo {photo_id}...")
        features['comments'] = []
        comments = self.flickr.photos.comments.getList(
            photo_id=photo_id).findall('*/comment')
        for comment in comments:
            features['comments'].append({
                'user': comment.get('author'),
                'comment': comment.text
            })
        self.tracker.increment('queries')
        # t1 = time.perf_counter()
        # self.logger.debug(f"comments: {t1-t0}")
        # t0 = t1

        self.logger.debug(
            f"About to get list of favorites of photo {photo_id}...")
        features['favorites'] = []
        person_walker = self.flickr.data_walker(
            self.flickr.photos.getFavorites,
            searchstring='*/person',
            photo_id=photo_id,
            per_page=_MAX_FAVORITES_PER_PAGE)
        for person in person_walker:
            features['favorites'].append(person.get('nsid'))
        queries_count = _count_walker_queries(
            len(features['favorites']), _MAX_FAVORITES_PER_PAGE)
        self.tracker.increment('queries', amount=queries_count)
        # t1 = time.perf_counter()
        # self.logger.debug(f"favorites: {t1-t0}")
        # t0 = t1

        self.logger.debug(f"About to get exif of photo {photo_id}...")
        features['exif'] = {}
        try:
            for entry in self.flickr.photos.getExif(photo_id=photo_id)[0]:
                features['exif'][entry.get('tag')] = entry[0].text
        except flickrapi.exceptions.FlickrError as e:
            if e.code == 2:
                self.logger.debug(
                    ("The owner of the photo "
                     f"{photo_id} "
                     "does not want to share EXIF data."))
            else:
                raise
        self.tracker.increment('queries')
        # t1 = time.perf_counter()
        # self.logger.debug(f"exif: {t1-t0}")
        # t0 = t1

        self.logger.debug(
            f"About to get list of groups of photo {photo_id}...")
        features['groups'] = []
        for pool in self.flickr.photos.getAllContexts(photo_id=photo_id).iter('pool'):
            features['groups'].append(pool.get('id'))
        self.tracker.increment('queries')
        # t1 = time.perf_counter()
        # self.logger.debug(f"groups: {t1-t0}")
        # t0 = t1

        self.logger.debug(
            f"Flickr features for photo {photo_id} collected.")
        return features

    def get_user_features(self, user_id):
        self.logger.debug(f"Starting to get features for user {user_id}")
        # t0 = time.perf_counter()
        features = {}

        person = self.flickr.people.getInfo(user_id=user_id)[0]
        features['nsid'] = person.get('nsid')
        features['ispro'] = (person.get('ispro') != '0')
        features['photo_count'] = person.find('.//count').text
        self.tracker.increment('queries')
        # t1 = time.perf_counter()
        # self.logger.debug(f"getInfo: {t1-t0}")
        # t0 = t1

        profile = self.flickr.profile.getProfile(user_id=user_id)[0]
        features['join_date'] = profile.get('join_date')
        features['occupation'] = profile.get('occupation')
        features['website'] = profile.get('website')
        features['profile_description'] = profile.get('profile_description')
        self.tracker.increment('queries')
        # t1 = time.perf_counter()
        # self.logger.debug(f"getProfile: {t1-t0}")
        # t0 = t1

        features['contacts'] = []
        contact_walker = self.flickr.data_walker(
            self.flickr.contacts.getPublicList,
            searchstring='*/contact',
            user_id=user_id,
            per_page=_MAX_CONTACTS_PER_PAGE)
        for contact in contact_walker:
            features['contacts'].append(contact.get('nsid'))
        queries_count = _count_walker_queries(
            len(features['contacts']), _MAX_CONTACTS_PER_PAGE)
        self.tracker.increment('queries', amount=queries_count)
        # t1 = time.perf_counter()
        # self.logger.debug(f"contacts: {t1-t0}")
        # t0 = t1

        features['groups'] = []
        for group in self.flickr.people.getPublicGroups(user_id=user_id)[0]:
            features['groups'].append(group.get('nsid'))
        self.tracker.increment('queries')
        # t1 = time.perf_counter()
        # self.logger.debug(f"groups: {t1-t0}")
        # t0 = t1

        return features

    def persist_features(self, features, path):
        with open(path, 'w') as jsonfile:
            json.dump(
                features,
                jsonfile,
                indent=4,
                ensure_ascii=False)

    def recover_features(self, path):
        with open(path) as f:
            return json.load(f)

    def _persist_photo_features(self, photo, photo_features_dir):
        photo_id = photo.get('id')

        photo_features_filepath = os.path.join(
            photo_features_dir, photo_id + ".json")
        if os.path.exists(photo_features_filepath):
            self.logger.debug(f"Backup for photo {photo_id} features found")
            self.tracker.increment('photos_cached')
        else:
            image_path = os.path.join(
                photo_features_dir, photo_id + '.' + self.image_format)
            _, img = self.download_photo(photo)
            if img is None:
                return False
            img = self.process_image(img)
            img.save(image_path)
            self.logger.debug(f"Photo {photo_id} downloaded to {image_path}")

            photo_features = self._insistent_call(
                self.get_basic_photo_features, photo_id)
            photo_features['width_o'] = photo.get('width_o')
            photo_features['height_o'] = photo.get('height_o')
            photo_features['width_downloaded'] = str(img.width)
            photo_features['height_downloaded'] = str(img.height)
            img.close()  # TODO Do it cleaner

            # Model scores
            self.logger.debug(f"Scoring photo {photo_id}...")
            for name, scorer in self.photo_scorers.items():
                photo_features[name + '_score'] = str(scorer.score(image_path))
            self.logger.debug(f"Photo {photo_id} scored.")

            os.remove(image_path)
            self.logger.debug(f"Photo {photo_id} removed from {image_path}")

            self.persist_features(photo_features, photo_features_filepath)
            self.logger.debug(f"Features for photo {photo_id} registered")
            self.tracker.increment('photos_registered')

            return True

    def sample_user_photos(self, user_id, per_page=500, add=False):
        left_requested = self.photo_sample_size
        if not add:
            photo_features_dir = os.path.join(
            self.output_path, 'user_features', user_id, 'photo_features')
            already_sampled_photos = {
                p.split('.')[0]
                for p in os.listdir(photo_features_dir)
                if p.endswith('.json')
            }
            left_requested -= len(already_sampled_photos)

        self.logger.debug(
            f"We proceed to the selection of {left_requested} photos...")

        rsp = self._insistent_call(
            self.flickr.photos.search,
            user_id=user_id,
            content_type=_CONTENT_TYPE,
            media=_MEDIA_TYPE,
            extras='license,url_o,' + ','.join(
                ['url_' + t for t in _SIZE_TYPES]),
            per_page=0)
        total_photos = int(rsp[0].get('total'))

        left_indexes = list(range(total_photos))
        random.seed(user_id)  # Reproducibility for faster error recovery
        while left_requested > 0:
            if not left_indexes:
                self.logger.warning(
                    (f"No more available photos for user {user_id}"))
                break
            photo_indexes = _sample_or_get_all(left_indexes, left_requested)
            photo_indexes.sort()

            old_page = 0
            for i in photo_indexes:
                self.logger.debug(f"Selected photo {i+1} of {total_photos}")
                # possible indexes from 0 to total_photos-1
                page = i // per_page + 1
                offset = i % per_page
                if page != old_page:
                    self.logger.debug(
                        f"Searching page={page} of {total_photos // per_page + 1}")
                    self.tracker.increment('queries')
                    # t0 = time.perf_counter()
                    rsp = self._insistent_call(
                        self.flickr.photos.search,
                        user_id=user_id,
                        content_type=_CONTENT_TYPE,
                        media=_MEDIA_TYPE,
                        extras='license,url_o,' + ','.join(
                            ['url_' + t for t in _SIZE_TYPES]),
                        per_page=per_page,
                        page=page)
                    photos = rsp.findall('*/photo')
                    # self.logger.debug(f"Time spent searching: {time.time()-t0}")

                    old_page = page

                try:
                    photo = photos[offset]
                except IndexError:
                    self.logger.warning(
                        (f"Photo {i+1} couldn't be selected. "
                        f"Probably deleted or private by user {user_id}"))
                else:
                    if add or photo.get('id') not in already_sampled_photos:
                        left_requested -= 1
                        yield photo

            left_indexes = [i for i in left_indexes if i not in photo_indexes]
            if left_requested > 0:
                self.logger.debug(
                    f"Reselecting photos to obtain the remaining {left_requested}...")


    def full_persist_user_and_photo_sample_features(self, user_id,
                                                    required_features=None,
                                                    add=False):
        user_features_dir = os.path.join(
            self.output_path, 'user_features', user_id)
        os.makedirs(user_features_dir, exist_ok=True)

        # Persist user features
        user_features_filepath = os.path.join(
            user_features_dir, user_id + ".json")
        if os.path.exists(user_features_filepath):
            self.logger.debug(f"Backup for user {user_id} features found")
            self.tracker.increment('users_cached')
        else:
            user_features = self._insistent_call(
                self.get_user_features, user_id)
            self.persist_features(user_features, user_features_filepath)

            self.logger.debug(f"User features for user {user_id} registered")
            self.tracker.increment('users_registered')

        # Discard user if doesn't have required features
        if required_features is not None:
            user_features = self.recover_features(user_features_filepath)
            for f in required_features:
                value = user_features.get(f)
                if (value is None) or (value == ''):
                    self.logger.info(
                        f"User {user_id} missing field {f} discarded")
                    return False, False

        # Photos
        #--------
        photo_features_dir = os.path.join(user_features_dir, 'photo_features')
        os.makedirs(photo_features_dir, exist_ok=True)

        # Sampling of the user photos
        photos = self.sample_user_photos(user_id, add=add)

        # Persist the features of every photo
        error_count_before = self.tracker.get_counter('error')
        for photo in photos:
            try:
                self._persist_photo_features(photo, photo_features_dir)
            except KeyboardInterrupt:
                raise
            except:
                self.logger.error(
                    f"Unexpected exception persisting photo {photo.get('id')} features", exc_info=True)
                self.tracker.increment('error')
            finally:
                self.tracker.track_progress()
        if (error_count_before is not None and
                error_count_before < self.tracker.get_counter('error')):
            return False, True

        self.logger.info(f"Done with user {user_id}")
        return True, True

    def get_activity_stats(self, min_upload_date, max_upload_date,
                           interval_length=300, per_page=500):
        params = {
            'min_upload_date': min_upload_date,
            'max_upload_date': min_upload_date + interval_length - 1,
            'sort': 'date-posted-asc',
            # photos only (exclude screenshots and "others")
            'content_type': 1,
            'media': 'photos',
            'extras': 'date_upload',
            'per_page': per_page
        }
        activity = {}
        while params['max_upload_date'] <= max_upload_date:
            page = 1
            pages = 1
            aux_min = params['min_upload_date']
            aux_max = params['max_upload_date']

            while page <= min(_MAX_N_RESULTS//per_page, pages):
                self.logger.debug(
                    f"Calling flickr.photos.search(page={page} of {pages}, {params})")
                rsp = self._insistent_call(
                    self.flickr.photos.search, page=page, **params)
                self.tracker.increment('queries')

                photoset = rsp.getchildren()[0]
                pages = int(photoset.get('pages'))
                total = int(photoset.get('total'))
                photos = rsp.findall('*/photo')
                self.logger.debug(
                    f"Total: {total} photos in {pages} pages")

                for photo in photos:
                    if activity.get(photo.get('owner')) is None:
                        activity[photo.get('owner')] = 1
                    else:
                        activity[photo.get('owner')] += 1

                    aux_min = min(int(photo.get('dateupload')), aux_min)
                    aux_max = max(int(photo.get('dateupload')), aux_max)
                self.logger.debug(
                    f"Minimum dateupload found until page {page}: {aux_min}")
                self.logger.debug(
                    f"Maximum dateupload found until page {page}: {aux_max}")

                page += 1

            self.logger.debug(
                f"Changing search parameters because: page={page}; pages={pages}")
            params['min_upload_date'] = aux_max + 1
            params['max_upload_date'] = aux_max + interval_length

        return activity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ffe-config-filepath',
                        help='file with flickr download configuration (default: config/ffe.conf)',
                        required=False,
                        default='config/ffe.conf')
    parser.add_argument('-l', '--log-config-filepath',
                        help='file with logging configuration (default: config/logging.conf)',
                        required=False,
                        default='config/logging.conf')
    args = parser.parse_args()

    logging.config.fileConfig(
        args.log_config_filepath,
        disable_existing_loggers=True)  # Only for debugging

    ffe = FlickrFeatureExtraction(args.ffe_config_filepath)
    with open(ffe.input_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row.get('user')
            ffe.full_persist_user_and_photo_sample_features(user_id)
