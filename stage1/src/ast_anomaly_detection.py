"""
# PS12: Underwater Domain Awareness
Stage1: ML inference using Mock datasets
"""

import os
import re

import numpy as np
import soundfile as sf

import json

from transformers import ASTConfig, ASTForAudioClassification

# mock test data dir path
testdir ="data/I4_20251013_PS12"
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

# list of audios file info
files = os.listdir(path_mock_testdir)
audios_list = [{} for i in range(len(files))]
annotations_list = []#[{} for i in range(len(files))]

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

        
    def detect_n_classify(self):
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
                elif (i < self.nbins - 1 and not class_id == nbins_classes[i+1]) or (i == self.nbins - 1):
                    # if len(self.start_times) > len(self.end_times):
                    self.end_times.append(np.floor((i + 1) * self.nsec / self.nbins))
                        
                    # classification
                    self.category_ids.append(int(class_id))
                    self.n_activity += 1

            if (i < self.nbins - 1) and nbins_classes[i+1] > 0:
                if not class_id == nbins_classes[i+1]:
                    self.start_times.append(np.floor((i + 1) * self.nsec / self.nbins))

            # classification
            # if np.random.uniform() > 0.5:
            #     c_id_factor = 2
            # else:
            #     c_id_factor = 4

            # # category id
            # self.category_ids.append(int(c_id_factor * np.random.uniform() + 1))


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
    d = {}
    for filename in files:
        # id
        f_id = int(re.search(r'_(\d+)', filename)[1])
        d[f_id] = filename
    sorted_filenames=sorted(d.items())

    for f_id, filename in sorted_filenames:
        # read wav file
        data, fs = sf.read(path_mock_testdir + "/" + filename)

        # sound duration in sec
        nsec = len(data)/fs
        
        # id
        # f_id = int(re.search(r'_(\d+)', filename)[1])
        audios_list[f_id-1] = {
            "id": f_id,
            "file_name": filename,
            "file_path": testdir + "/" + filename,
            "duration": nsec
        }
        
        # Audio_Detect_Classify class instance
        sound_wave = Audio_Detect_Classify(f_id, data, fs, nsec, nbins=8)

        # sound detection & classification
        sound_wave.detect_n_classify()

        # update annotations
        sound_wave.update_annotations()

    # save
    save_output('2025_10_13_AIRTRAIL_PS12_S1_MOCK_I4.json')


if __name__=="__main__":
    main()