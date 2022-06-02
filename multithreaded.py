import argparse
import configparser
import gc
import logging
import os
import queue
import sys
import threading
import tracemalloc

import flickrapi
import pandas as pd
import tensorflow as tf

# https://github.com/tensorflow/tensorflow/issues/29968
os.environ['OMP_NUM_THREADS'] = '2'
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

import tracking
from flickr_feature_extraction import FlickrFeatureExtraction
from nima import nima
from shukong_aesthetics import shukong_aesthetics


AUX_METADATDA_PATH = '.metadata_updates.csv'


class ThreadSafeScorerAdapter():

    def __init__(self, scorer, name=None):
        self.lock = threading.Lock()
        self.scorer = scorer
        if name is None:
            self.scorer_name = self.scorer.__class__.__name__
        else:
            self.scorer_name = name

    def score(self, image_path):
        # logging.debug(
        #     f"Thread {threading.current_thread().name} acquiring {self.scorer_name} lock...")
        self.lock.acquire()
        # logging.debug(
        #     f"{self.scorer_name} lock acquired by thread {threading.current_thread().name}")
        score = self.scorer.score(image_path)
        # logging.debug(
        #     f"{self.scorer_name} lock released by thread {threading.current_thread().name}")
        self.lock.release()
        return score


class FlickrFeatureExtractionMultithreaded():

    class Worker(threading.Thread):
        def __init__(self, name, ffe, pending_tasks_queue, finished_tasks_queue):
            self.pending_tasks = pending_tasks_queue
            self.finished_tasks = finished_tasks_queue
            self.ffe = ffe
            self.logger = logging.getLogger(__name__)
            threading.Thread.__init__(self, name=name, daemon=True)
            self.start()

        def run(self):
            while True:
                task_name, func, args, kwargs = self.pending_tasks.get()
                has_exception = False
                retvalues = None
                s1 = tracemalloc.take_snapshot()
                try:
                    retvalues = func(self.ffe, *args, **kwargs)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    has_exception = True
                    retvalues = e
                    self.logger.error(
                        (f"Thread {self.name} encountered an exception while "
                         "doing the task "
                         f"'{task_name}': {func.__name__}({args}, {kwargs})"),
                        exc_info=True)
                else:
                    self.logger.debug(
                        (f"Thread {self.name} finished with task "
                         f"'{task_name}': {func.__name__}({args}, {kwargs})"))
                    if retvalues[1]:
                        s2 = tracemalloc.take_snapshot()
                        self.logger.debug(
                            f"{gc.collect()} unreachable objects from garbage collector")
                        top_stats = s2.compare_to(s1, "lineno")[:10]
                        print('Snapshot comparation for task',
                              task_name, file=sys.stderr)
                        for s in top_stats:
                            print(s, file=sys.stderr)
                        print('\n\n\n', file=sys.stderr)
                finally:
                    self.finished_tasks.put(
                        (task_name, has_exception, retvalues))
                    self.pending_tasks.task_done()

    def __init__(self, config_filepath, *secrets_filepaths):
        self.workers = []
        self.pending_tasks = queue.Queue()
        self.finished_tasks = queue.SimpleQueue()
        self.logger = logging.getLogger(__name__)
        photo_scorers = {
            'kong': ThreadSafeScorerAdapter(
                shukong_aesthetics.ShuKongAestheticScorer(), name='ShuKong'),
            'nima': ThreadSafeScorerAdapter(
                nima.NimaScorer(tech=False), name='NimaAesthetic'),
            'nima_tech': ThreadSafeScorerAdapter(
                nima.NimaScorer(tech=True), name='NimaTechnical')
        }
        tracker = tracking.ThreadSafeAdapter(tracking.Tracker())

        for p in secrets_filepaths:
            ffe = FlickrFeatureExtraction(
                config_filepath, p, photo_scorers, tracker)
            name = os.path.basename(p).split('.')[0]
            self.workers.append(
                FlickrFeatureExtractionMultithreaded.Worker(
                    name, ffe, self.pending_tasks, self.finished_tasks))

    def add_task(self, task_name, func, *args, **kwargs):
        """ Add a task to the queue """
        self.pending_tasks.put((task_name, func, args, kwargs))

    def map(self, tasks_names, func, args_list, **kwargs):
        """ Add a list of tasks to the queue """
        for task_name, args in zip(tasks_names, args_list):
            self.add_task(task_name, func, *args, **kwargs)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.pending_tasks.join()


if __name__ == "__main__":
    tracemalloc.start()

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

    ffe = FlickrFeatureExtractionMultithreaded(
        args.ffe_config_filepath,
        'original.env', 'a1.env', 'a2.env', 'b1.env', 'b2.env')

    config_parser = configparser.ConfigParser()
    config_parser.read(args.ffe_config_filepath)
    input_path = config_parser['IO']['input_path']
    df = pd.read_csv(input_path, index_col='user')
    users_left = df.query(
        'completed==False and has_occupation!=False and deleted!=True').index

    ffe.map(
        users_left,
        FlickrFeatureExtraction.full_persist_user_and_photo_sample_features,
        users_left.values.reshape(-1, 1),
        required_features=['occupation'])

    try:
        for _ in range(len(users_left)):
            user_id, has_exception, retvalues = ffe.finished_tasks.get()
            if has_exception:
                is_completed = False
                has_occupation = 'NA'
                is_deleted = 'NA'
                if isinstance(retvalues, flickrapi.exceptions.FlickrError):
                    is_deleted = retvalues.code == 5
            else:
                is_completed = retvalues[0]
                has_occupation = retvalues[1]
                is_deleted = False
            with open(AUX_METADATDA_PATH, 'a') as aux:
                row = ','.join(
                    map(str, [user_id, is_completed, has_occupation, is_deleted]))
                aux.write(row + '\n')
    except KeyboardInterrupt:
        logging.info("Extraction interrupted by the user.")
    else:
        logging.info("Extraction finished!!!!!!!!!! Oleee!")
    finally:
        df_aux = pd.read_csv(
            AUX_METADATDA_PATH,
            header=None,
            names=[df.index.name] + df.columns.tolist(),
            index_col='user')
        df.loc[df_aux.index] = df_aux
        df.to_csv(input_path)
        os.remove(AUX_METADATDA_PATH)
