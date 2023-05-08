<p align="center"><img width="160" src="doc/lip_white.png" alt="logo"></p>
<h1 align="center">Visual Speech Recognition for Multiple Languages</h1>

<div align="center">

[📘Introduction](#Introduction) |
[🛠️Preparation](#Preparation) |
[📊Benchmark](#Benchmark-evaluation) |
[🔮Inference](#Speech-prediction) |
[🐯Model zoo](#Model-Zoo) |
[📝License](#License)
</div>

## Introduction

This is LipReading Project which decode the movement of the voice-impaired's mouth to understand what they want to say without speaking out loud. We use the model from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages that achieve the performance of 19.1% WER for visua speech recognition (VSR) on LRS3. The model is served and deployed in Torchserve

## Preparation

1. Setup the environment.
```Shell
conda create -y -n autoavsr python=3.8
conda activate autoavsr
```

2. Install pytorch, torchvision, and torchaudio by following instructions [here](https://pytorch.org/get-started/), and install all packages:

```Shell
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

3. Download and extract a pre-trained model and/or language model from https://drive.google.com/drive/folders/1CIpUHZ3teumOgH7MukNJVH9gmikWGikb?usp=sharing to:

- `./benchmarks/${dataset}/models`

- `./benchmarks/${dataset}/language_models`

4. [For VSR and AV-ASR] Install [RetinaFace](./tools) or [MediaPipe](https://pypi.org/project/mediapipe/) tracker.

### Benchmark evaluation

```Shell
python eval.py config_filename=[config_filename] \
               labels_filename=[labels_filename] \
               data_dir=[data_dir] \
               landmarks_dir=[landmarks_dir]
```

- `[config_filename]` is the model configuration path, located in `./configs`.

- `[labels_filename]` is the labels path, located in `${lipreading_root}/benchmarks/${dataset}/labels`.

- `[data_dir]` and `[landmarks_dir]` are the directories for original dataset and corresponding landmarks.

- `gpu_idx=-1` can be added to switch from `cuda:0` to `cpu`.

### Speech prediction

```Shell
python infer.py config_filename=[config_filename] data_filename=[data_filename]
```

- `data_filename` is the path to the audio/video file.

- `detector=mediapipe` can be added to switch from RetinaFace to MediaPipe tracker.

### Mouth ROIs cropping

```Shell
python main.py data_filename=[data_filename] dst_filename=[dst_filename]
```

- `dst_filename` is the path where the cropped mouth will be saved.

### Torchserve
1. Install Torchserve
```Shell
pip install torchserve torch-model-archiver torch-workflow-archiver
```
2. Install nvgpu for torchserve
```Shell
pip install nvgpu
```
3. Download the LipReading.mar in https://drive.google.com/file/d/1Wpr9CiWqdaVHcPv8FXCu-nhm82Bb00E0/view?usp=sharing and put it in model_store folder. Start Torchserve
```Shell
torchserve --start --ncs --model-store model_store --models LipReading.mar
```
4. Test API
```Shell
curl -d data=[path_to_datafile] http://127.0.0.1:8080/predictions/LipReading
```


## Citation

```bibtex
@inproceedings{ma2023auto,
  author={Ma, Pingchuan and Haliassos, Alexandros and Fernandez-Lopez, Adriana and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels}, 
  year={2023},
}
```

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

