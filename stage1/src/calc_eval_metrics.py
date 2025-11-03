"""
# PS12: Underwater Domain Awareness
Calculate the evaluation metrics (IER) w/ ground truth labels
Author: mandeeps2509@gmail.com
"""

import os

from pydub import AudioSegment

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import json

import argparse

# weights for IER calculation
W = {"f": 0.25, "m": 1, "c": 0.75}

def get_list_labels(path_annotations, dir_path, n_chunks=16):
    with open(path_annotations) as file:
        annotations_data = json.load(file)
    # dataframe of audio files list and its annotated categories
    df_audios = pd.DataFrame(annotations_data['audios'])
    df_annotations = pd.DataFrame(annotations_data['annotations'])

    # list of audio chunks filenames & its labels
    audio_chunks_labels = []
    # sorted list of files
    sorted_files = sorted(os.listdir(dir_path))
    for filename in sorted_files:
        if filename.endswith(('wav', 'mp3')):
            data = AudioSegment.from_wav(dir_path + "/" + filename)

            # len of chunk (in ms)
            chunk_len = len(data)/n_chunks

            # extract labels from annotated data
            labels = [0 for _ in range(n_chunks)]

            # audio_id for filename
            # if not df_audios['file_name'].str.endswith("wav", "mp3"):
            try:
                audio_id = df_audios[df_audios['file_name']==filename[:-4]]['id'].iloc[0]
            except:
                audio_id = df_audios[df_audios['file_name']==filename]['id'].iloc[0]

            # labels
            curr_annotations = df_annotations[df_annotations['audio_id']==audio_id]
            for ind in range(len(curr_annotations)):
                # print(f"audio_id: {audio_id}, {curr_annotations.iloc[ind]['start_time']}s - {curr_annotations.iloc[ind]['end_time']}s")
                i_start = round(curr_annotations.iloc[ind]['start_time']* 1000 / chunk_len) # start index of chunk
                i_end = round((curr_annotations.iloc[ind]['end_time'] - curr_annotations.iloc[ind]['start_time']) * 1000 / chunk_len) + i_start - 1
                # print(f"audio_id: {audio_id}, i_start: {i_start} ({curr_annotations.iloc[ind]['start_time']/chunk_len}), i_end: {i_end} ({curr_annotations.iloc[ind]['end_time']/chunk_len})")
                for j in range(i_start, i_end + 1):
                    labels[j] = int(curr_annotations.iloc[ind]['category_id'])
                    
            audio_chunks_labels = audio_chunks_labels + labels
    return audio_chunks_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--ground_truth", default="./datasets/gt_annotations.json")
    parser.add_argument("-p", "--prediction", default="./results/output.json")
    parser.add_argument("-d", "--dir_path", default="./data/20251006_PS12")
    parser.add_argument("-n", "--n_chunks", default=16)
    args = parser.parse_args()

    # get list of ground truth labels from json (annotations format)
    b_gt_list = False
    try:
        list_gt_labels = get_list_labels(args.ground_truth, args.dir_path, int(args.n_chunks))
        print(f"====> Loaded ground truth labels from {args.ground_truth}")
        b_gt_list = True
    except:
        print(f"****** Ground truth labels not present at {args.ground_truth}")

    # get list of predicted labels from json (annotations format)
    b_pred_list = False
    try:
        list_pred_labels = get_list_labels(args.prediction, args.dir_path, int(args.n_chunks))
        print(f"====> Loaded prediction labels from {args.prediction}")
        b_pred_list = True
    except:
        print(f"****** Prediction labels not present at {args.prediction}")

    if b_gt_list and b_pred_list:
        # dataframe with combined gt & prediction labels
        df_metric = pd.DataFrame({'gt': list_gt_labels, 'pred': list_pred_labels})

        # calculate confusion matrix
        Confusion = confusion_matrix(df_metric['gt'], df_metric['pred'], labels=[0, 1, 2, 3, 4])
        print(f"Confusion matrix:\n {Confusion}")

        # Identification Error Rate (IER)
        ier = W['f'] * Confusion[0][1:].sum()   # false alarm
        ier += W['m'] * Confusion[1:][0].sum()  # missed acoustic events
        ier += W['c'] * (Confusion[1:][1:].sum() - np.trace(Confusion[1:][1:]))   # class confusion
        ier = ier / Confusion.sum()
        print(f">>>> Identification Error Rate (IER): {ier}")

if __name__=="__main__":
    main()