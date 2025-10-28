"""
# PS12: Underwater Domain Awareness
Stage1: ML inference using Mock/pre-eval datasets
Author: mandeeps2509@gmail.com
"""

import os
import re

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import make_chunks

import json
import argparse
import pandas as pd

from transformers import ASTConfig, ASTForAudioClassification
from datasets import Dataset, Audio, ClassLabel, Features

from fine_tune_ast import fine_tune_AST

# mock test data dir path
data_dir = "data/2025_09_12_PS_12"
# testdir ="data/I4_20251013_PS12"
testdir = "data/pre-eval/20251027_PS12"
path_mock_testdir = os.getcwd() + "/" + testdir

# category mapping
categories = [{"id": 1, "name": "vessel"},
              {"id": 2, "name": "marine_animal"},
              {"id": 3, "name": "natural_sound"},
              {"id": 4, "name": "other_anthropogenic"}]

# pre-trained model name
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"

# load configuration for pre-trained AST model
config = ASTConfig.from_pretrained(pretrained_model)

# config update: id2label as per mapping
config.num_labels = len(categories)
config.id2label = {i: category["name"] for i, category in enumerate(categories)}
config.label2id = {v: k for k, v in config.id2label.items()}

audio_keys = ["id", "file_name", "file_path", "duration"]
info_dict = {"description": "Grand Challenge UDA", 
             "version": "1.0",
             "year": 2025}

def export_audio_chunks(dir_path, n_chunks, b_labels=False, path_annotations=None, b_save_dict=True):
    # export n_chunks of audio segments from audio dataset
    if os.path.isdir(dir_path):
        if not os.path.exists(dir_path + "/chunks"):
            os.mkdir(dir_path + "/chunks")

        if b_labels:
            with open(path_annotations) as file:
                annotations_data = json.load(file)
            # dataframe of audio files list and its annotated categories
            df_audios = pd.DataFrame(annotations_data['audios'])
            df_annotations = pd.DataFrame(annotations_data['annotations'])

        # dict of audio chunks filenames & its labels
        dict_audio_chunks_labels = {'audio': [], 'labels': []}
        # sorted list of files
        sorted_files = sorted(os.listdir(dir_path))
        for filename in sorted_files:
            if filename.endswith(('wav', 'mp3')):
                data = AudioSegment.from_wav(dir_path + "/" + filename)

                # len of chunk (in ms)
                chunk_len = len(data)/n_chunks

                # extract labels from annotated data
                labels = [0 for _ in range(n_chunks)]
                if b_labels:
                    # audio_id for filename
                    audio_id = df_audios[df_audios['file_name']==filename[:-4]]['id'].iloc[0]
                    # labels
                    curr_annotations = df_annotations[df_annotations['audio_id']==audio_id]
                    for ind in range(len(curr_annotations)):
                        # print(f"audio_id: {audio_id}, {curr_annotations.iloc[ind]['start_time']}s - {curr_annotations.iloc[ind]['end_time']}s")
                        i_start = round(curr_annotations.iloc[ind]['start_time']* 1000 / chunk_len) # start index of chunk
                        i_end = round((curr_annotations.iloc[ind]['end_time'] - curr_annotations.iloc[ind]['start_time']) * 1000 / chunk_len) + i_start - 1
                        # print(f"audio_id: {audio_id}, i_start: {i_start} ({curr_annotations.iloc[ind]['start_time']/chunk_len}), i_end: {i_end} ({curr_annotations.iloc[ind]['end_time']/chunk_len})")
                        for j in range(i_start, i_end + 1):
                            labels[j] = int(curr_annotations.iloc[ind]['category_id'])
                            # print(f'[dbg] labels[{j}]: {labels[j]}')
                    # print(f'***[dbg]*** filename {filename}, audio_id: {audio_id}, labels: {labels}')
                dict_audio_chunks_labels["labels"] = dict_audio_chunks_labels["labels"] + labels

                # n_chunks audio segment 
                chunks=make_chunks(data, chunk_len)
                # list of audio chunk filenames
                audio_chunk_names = []
                for i, chunk in enumerate(chunks):
                    chunk_name=filename[:-4] + f"-chunk-{i}.wav"
                    chunk.export(dir_path + "/chunks/" + chunk_name, format="wav")
                    audio_chunk_names.append(dir_path + "/chunks/" + chunk_name)
                dict_audio_chunks_labels["audio"] = dict_audio_chunks_labels["audio"] + audio_chunk_names      

        if b_save_dict:
            # save dict
            with open(dir_path + "/chunks/dict_audio_chunks_labels.json", 'w+') as file:
                json.dump(dict_audio_chunks_labels, file)
        return True, dict_audio_chunks_labels
    else:
        return False, None    

def init_files_ds(path_testdir):
    global files, audios_list, annotations_list
    # list of audios file info
    if not os.path.isdir(path_testdir):
        return False
    else:
        files = os.listdir(path_testdir)
        audios_list = [{} for i in range(len(files))]
        annotations_list = []#[{} for i in range(len(files))]
        return True

def init_model(model_name):
    global model
    # class instance of model
    if model_name.lower() == 'ast':
        model = fine_tune_AST([{"id": 0, "name": "noise"}] + categories)
        return True
    else:
        return False

class Audio_Detect_Classify():
    def __init__(self, f_id, data, fs, nsec, nbins=10):
        # audio file id
        self.f_id = f_id
        # duration
        self.nsec = nsec
        # normalize signal 
        self.mean = data.mean()
        self.std = data.std()

        self.XY = np.empty((len(data), 2), float)
        self.XY[:,0] = np.array([i/fs for i in range(len(data))])

        # 0 mean & 0.5 std
        self.XY[:,1] = 0.5 * (data - self.mean) / self.std
        
        # # of bins
        self.nbins = nbins

        
    def detect_n_classify(self, model):
        # detection
        # prob distribution of categories
        probs = [0.5, 0.5*0.5*0.75, 0.5*0.5*0.75, 0.5*0.5*0.25, 0.5*0.5*0.25]

        # detect & classify
        rng = np.random.default_rng()
        nbins_classes = rng.choice([0, 1, 2, 3, 4], size=self.nbins, p=probs)

        # total sound activity
        self.n_activity = 0

        self.start_times = []
        self.end_times = []
        self.category_ids = []
        for i, class_id in enumerate(nbins_classes):
            if class_id > 0:
                if i == 0:
                    self.start_times.append(0.0)
                if (i < self.nbins - 1 and not class_id == nbins_classes[i+1]) or (i == self.nbins - 1):
                    # if len(self.start_times) > len(self.end_times):
                    self.end_times.append(np.floor((i + 1) * self.nsec / self.nbins))
                        
                    # classification
                    self.category_ids.append(int(class_id))
                    self.n_activity += 1

            if (i < self.nbins - 1) and nbins_classes[i+1] > 0:
                if not class_id == nbins_classes[i+1]:
                    self.start_times.append(np.floor((i + 1) * self.nsec / self.nbins))

            # overlapped classes check
            self.check_overlapped_classes()
            if self.b_overlapped_classes:
                print(f"=====> Overlapped classes in file #{self.f_id}")

            # classification
            # if np.random.uniform() > 0.5:
            #     c_id_factor = 2
            # else:
            #     c_id_factor = 4

            # # category id
            # self.category_ids.append(int(c_id_factor * np.random.uniform() + 1))

    def check_overlapped_classes(self):
        self.b_overlapped_classes = False
        for i in range(self.n_activity - 1):
            if self.end_times[i] > self.start_times[i + 1]:
                self.b_overlapped_classes = True
                return

    def update_annotations(self):
        # annotations
        id = len(annotations_list)
        for i in range(self.n_activity):
            annotations = {
                "id": id + i + 1, 
                "audio_id": self.f_id, 
                "category_id": self.category_ids[i],
                "start_time": self.start_times[i],
                "end_time": self.end_times[i],
                "duration": self.end_times[i] - self.start_times[i],
                "score": np.random.uniform()
            }
            # print(f'<<<<<Audio #{self.f_id} annotations>>>>>')
            # print(annotations)
            annotations_list.append(annotations)


def save_output(filename='2025_10_06_AIRTRAIL_PS12_S1_MOCK3.json'):
    output_json = {
        "info": info_dict,
        "audios": audios_list,
        "categories": categories,
        "annotations": annotations_list
    }

    # print("\n ****** FINAL Annotations list ****** \n", annotations_list)

    with open(filename, 'w') as f:
        json.dump(output_json, f)


def main():
    # get val to arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", default=False)
    parser.add_argument("-trd", "--train_dir", default=data_dir)
    parser.add_argument("-tra", "--train_audiolabels", default=None)
    parser.add_argument("-m", "--model", default="ast")
    parser.add_argument("-n", "--nbins", default=16)
    parser.add_argument("-t", "--test", default=True)
    parser.add_argument("-td", "--test_dir", default=testdir)
    parser.add_argument("-o", "--output_file", default='2025_10_27_AIRTRAIL_PS12_S1_PRE-EVAL.json')#'check-2025_10_13_AIRTRAIL_PS12_S1_MOCK_I4.json')
    args = parser.parse_args()

    global model
    b_init = init_model(args.model)
    if b_init and args.train:
        # export chunks of dataset
        if args.train_audiolabels is None:
            b_audio_chunks, dict_audios_labels = export_audio_chunks(os.getcwd() + "/" + args.train_dir, 
                                                    args.nbins,
                                                    b_labels=True, 
                                                    path_annotations=os.getcwd() + "/" + args.train_dir + "/gt_annotations.json")
            print('===> Exported audio chunks')
        else:
            b_audio_chunks = True
            with open(args.train_audiolabels, 'r') as file:
                dict_audios_labels = json.load(file)
            print('===> Loaded audio chunks')

        if b_audio_chunks:
            # dataset from dict of audios & labels
            dataset = Dataset.from_dict(
                dict_audios_labels,
                features=Features({
                    'audio': Audio(),
                    'labels': ClassLabel(names=['noise'] + [x['name'] for x in categories])
                })
            )
        # Apply the transformation to the dataset
        dataset = dataset.rename_column("audio", "input_values")  # rename audio column
        dataset.set_transform(model.preprocess_audio, output_all_columns=False)

        # split training data
        if "test" not in dataset:
            dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=0, 
                                               stratify_by_column="labels")

        print("====> Dataset loaded and pre-processed for training")
        model.training(dataset, batch_size=args.nbins)
         
    if b_init and args.test:
        if init_files_ds(os.getcwd() + "/" + args.test_dir):
            global files
            d = {}
            for filename in files:
                # id
                f_id = int(re.search(r'_(\d+)', filename)[1])
                d[f_id] = filename
            sorted_filenames=sorted(d.items())
            print(">> Entered test mode:")

            for f_id, filename in sorted_filenames:
                # read wav file
                data, fs = sf.read(os.getcwd() + "/" + args.test_dir + "/" + filename)

                # sound duration in sec
                nsec = len(data)/fs
                
                # id
                # f_id = int(re.search(r'_(\d+)', filename)[1])
                audios_list[f_id-1] = {
                    "id": f_id,
                    "file_name": filename,
                    "file_path": args.test_dir + "/" + filename,
                    "duration": nsec
                }
                
                # Audio_Detect_Classify class instance
                sound_wave = Audio_Detect_Classify(f_id, data, fs, nsec, nbins=args.nbins)

                # sound detection & classification
                sound_wave.detect_n_classify(model)

                # update annotations
                sound_wave.update_annotations()

            # save
            save_output(args.output_file)
            print(f'===> Saved output file: {args.output_file}')

if __name__=="__main__":
    main()