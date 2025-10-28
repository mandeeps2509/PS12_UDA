# AI Grand Challenge - 2025
## Underwater Domain Awareness
#### Author: Mandeep Singh (mandeeps2509@gmail.com)

### Stage1:
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
- Use Dockerfile to build the image, UDA_IMAGE with tag v1 (as per build-arg values):
```
docker build -f docker/Dockerfile --build-arg PY_VER=3.9 --build-arg BACKEND=mps --build-arg USER_UID=12345 --build-arg USERNAME=docker_usr --build-arg USER_GID=12345 --build-arg GROUPNAME=docker_ml -t UDA_IMAGE:v1 .
```
- Check the build docker image:
```
docker image ls
```
- Run the docker with container name: UDA_CONTAINER at port 8080:80
```
docker run -d --name UDA_CONTAINER -p 8080:80 UDA_IMAGE:v1
```
- Check the running status of the docker container
```
docker ps
```

#### [Inference]
- Test mode only w/ test data at /data/test & output results in output.json:
```
python3 ast_anomaly_detection.py -tr=False -t=True -td='data/test' -o='data/output.json'
```