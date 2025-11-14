# AI Grand Challenge - 2025
## Underwater Domain Awareness
#### Author: Mandeep Singh (mandeeps2509@gmail.com)

### [Stage1]
- Dockerised solution for ML detection & inference 

#### Model selection & training 
- Pre-trained Audio Spectrogram Transformer (AST) model is fine-tuned with the training datasets with 5 classes: 
```
{
    "0": "noise",
    "1": "vessel",
    "2": "marine_animal",
    "3": "natural_sound",
    "4": "other_anthropogenic"
}
```
- Datasets can be placed in './data' directory
- *NOTE*: Final output.json (annotations) doesn't contain audio segment(s) for {"0": "noise"} class

#### [Source_code]
Following are the input arguments required by src code to run: 
- ast_anomaly_detection.py: 
    ("-tr", "--train", default=False, type=bool)
    ("-trd", "--train_dir", default=data_dir)
    ("-tra", "--train_audiolabels", default=None)
    ("-lr", "--learning_rate", default=1e-5)
    ("-ep", "--num_epoch", default=15, type=int)
    ("-ft", "--peft", default=False, type=bool)
    ("-ftd", "--peft_dir", default="./trained_ast_clf_lora")
    ("-m", "--model", default="ast")
    ("-lm", "--load_model", default=True, type=bool)
    ("-md", "--model_dir", default="./trained_ast_clf")
    ("-n", "--nbins", default=16, type=int)
    ("-t", "--test", default=True, type=bool)
    ("-td", "--test_dir", default=testdir)
    ("-o", "--output_file", default='2025_10_27_AIRTRAIL_PS12_S1_PRE-EVAL.json')
- calc_eval_metrics.py:
    ("-gt", "--ground_truth", default="./datasets/gt_annotations.json")
    ("-p", "--prediction", default="./results/output.json")
    ("-d", "--dir_path", default="./data/20251006_PS12")
    ("-n", "--n_chunks", default=16)

#### [Training]
- Training without audio chunks from the training data (raw data), no test mode:
```
python3 ast_anomaly_detection.py -tr=True -trd='data/train' -t=False
```
- Training with audio chunks from the training data {w/ annotated labels: dict_audio_chunks_labels.json}
```
python3 ast_anomaly_detection.py -tr=True --trd='data/train' -tra='data/train/chunks/dict_audio_chunks_labels.json' -t=False
```
>> Model is saved at './trained_ast_clf'

#### [Setup]
*Dockerfile uses requirements-core.txt & requirements-extend.txt to build docker image in multi-stages*
*NOTE*: 1. Check for the installation of ffmpeg(>=6.1) & torchcodec that is required
        2. Update args values for src/ast_anomaly_detection.py in docker/entrypoint.sh
        3. Ensure host HW architecture in Dockerfile (here, aarch64)

- Use Dockerfile to build the image, UDA_IMAGE with tag v1 (as per build-arg values; ensure the correct torchcodec's gzip URL, here v0.7.0):
```
docker build -f docker/Dockerfile --build-arg PY_VER=3.9 --build-arg BACKEND=mps --build-arg USER_UID=12345 --build-arg USERNAME=docker_usr --build-arg USER_GID=12345 --build-arg GROUPNAME=docker_ml --build-arg TORCHCODEC_URL="https://github.com/meta-pytorch/torchcodec/archive/refs/tags/v0.7.0.tar.gz" -t UDA_IMAGE:v1 .
```
- Check the build docker image:
```
docker image ls
```
- Run the docker with container name: UDA_CONTAINER at port 8080:80 w/ volume mount ./data at /ml_app/data in UDA_CONTAINER
```
docker run -it -p 8080:80 -v "$(pwd)"/data:/ml_app/data:ro --name UDA_CONTAINER UDA_IMAGE:v1 /bin/bash
```
- Check the running status of the docker container
```
docker ps
```
- Copy entrypoint.sh's python script output files {-o} to host in ./docker/run, if any:
```
docker container cp UDA_CONTAINER:/ml_app/results/output.json ./docker/run
```
- Remove container using CONTAINER_ID from CMD>> docker ps -a
```
docker rm CONTAINER_ID
```

#### [Inference]
- Test mode only w/ test data at ../data/test & output results in ./results/output.json:
```
python3 ast_anomaly_detection.py -tr=False -t=True -td='../data/test' -o='./results/output.json'
```

#### [Evaluate]
- Calculate IER using ground truth annotations 20251007_I3_gt_annotations.json:
```
python3 calc_eval_metrics.py -gt "../data/mock/20251007_I3_gt_annotations.json" -p "./results/20251006_I3_lora_n32_output.json" -d "../data/mock/20251006_PS12" -n 32
```