# **CONTRASTIVEFLOW-TTS: Enhancing Flow Matching for TTS with Contrastive Representation Learning**

This repository contains the implementation of **CONTRASTIVEFLOW-TTS**, which aims to enhances TTS systems by integrating contrastive representation learning into flow matching techniques.


Follow these steps to set up the environment for this project:
## Environment Setup
# **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-project-name.git
   cd your-project-name
   ```
# Install required packages
    ```bash
    conda create -n your_env_name python=3.9
    conda activate your_env_name
    pip install -r requirements.txt
    ```

# Set Environment Variables
    ```bash
    source path.sh  # change the env name in it if you don't use "CFTTS"
    ```
# Compile Extensions
    ```bash
    cd model/monotonic_align
    python setup.py build_ext --inplace
    ```

## Data Preparation
Organize your data in a Kaldi-style directory structure, place all your data description files in data/your_dataset/. Ensure you have the following files:
`wav.scp`: Maps utterance IDs to audio file paths
`utts.list`: A list of utterance IDs. Generate it with:
`utt2spk`: Maps each utterance to a speaker ID.
`text`: Maps each utterance to a speaker ID.
`phn_duration`: Lists the duration of each phoneme in frames.
Create `data/your_dataset/phones.txt` to map phonemes to integer indices.


## Extract Features
```shell
bash extract_fbank.sh --stage 0 --stop_stage 2 --nj 16
# nj: number of parallel jobs. 
```
We by default use 16kHz features here
This will create `feats/fbank` and `feats/normed_fbank`.

We also need to extract huBERT embedding in advance if training using
`utils/hubert_embedding_extraction.py`

The generated embedding will be saved in `data/ljspeech/train_hubert`

## Training
Configurations for training is stored as yaml file in `configs/`.

Then, training is performed by 
```shell
python train.py -c configs/${your_yaml} -m ${model_name}

```

## Generate Data for Rectified Flow Training
Use the trained model to generate new training data:
```shell
python generate_for_reflow.py -c configs/${your_yaml} -m ${model_name} \
                              --EMA --max-utt-num 100000000 \
                              --dataset train \
                              --solver euler -t 10 \
                              --gt-dur
```

Generated data will be saved in `synthetic_wav/your_model_name/generate_for_reflow/train/`

## Retrain the Model with Rectified Flow data

Create a new configuration file (e.g., configs/your_reflow_config.yaml) and update the data paths:

```yaml
perform_reflow: true
...
data:
    train:
        feats_scp: "synthetic_wav/{config_name}/train/feats.scp"
        noise_scp: "synthetic_wav/{config_name}/train/noise.scp"
...
```

## Retrain the Model with Rectified Flow data
```shell
python train.py -c configs/your_reflow_config.yaml -m your_reflow_model_name
```


## Inference
To synthesize speech using the trained model:
```shell
python inference_dataset.py -c configs/${your_yaml} -m ${model_name} --EMA \
                          --solver euler -t 10
```
Adjust parameters like --solver and -t (timesteps) as needed
Synthesized mel-spectrograms will be saved in 'synthetic_wav/your_model_name/tts_gt_spk/' The feats.scp file lists the generated features.

Change from acoustic features to wav files can then be done in the `hifigan/` directory. Follow the instructions in the hifigan/ directory or use your preferred vocoder.

## Acknowledgement
This project is inspired by and builds upon the following works:
* [VoiceFlowTTS](https://github.com/X-LANCE/VoiceFlow-TTS) 
* [GradTTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)
* [VITS](https://github.com/jaywalnut310/vits)
* [HuBERT](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert)
* [CPC](https://github.com/davidtellez/contrastive-predictive-coding)

