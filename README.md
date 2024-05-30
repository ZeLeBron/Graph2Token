# Graph2Token

## Requirements

* conda create -n molca python=3.8
* conda activate molca
* conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
* conda install pyg -c pyg
* pip install rouge_score nltk ogb peft rdkit salesforce-lavis
* pip install -U transformers pytorch-lightning
* pip install deepspeed
* Download nltk corpus:
```bash
import nltk

nltk.download('wordnet')
```

## Dataset

* Unzip qm9.zip under the `./data/` directory.

## Train
**Fine-tune Stage.** Run the following script for fine-tuning on the PubChem324k dataset:
Please download the checkpoints of vicuna 1.5 from this [link](https://huggingface.co/lmsys/vicuna-7b-v1.5) and put them under the `./LLM` directory.

```bash
python main_graph_token_regression.py
```