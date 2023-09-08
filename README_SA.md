# **S-Align (Soft alignment for E2E Speech Translation)**

The code is forked from Fairseq-v0.12.3. For more Installation details, please refer to [`Fairseq`](https://github.com/facebookresearch/fairseq/tree/v0.12.2) 

## Useage

Training scripts and configurations for the MuST-C dataset are as follows:

```
egs
|---machine_translation
|    |---train.sh
|    |---decode.sh
|    |---load_embedding.py
|---pretrain-all
|    |---joint_train_merge.sh
|    |---decode.sh
|    |---device_run.sh
|    |---conf
```
### Step 1. MT Pretrain

&bull; Prepare MT training data.

&bull; Modify the necessary paths in `machine_translation/train.sh`, and run `machine_translation/train.sh` to pretrain MT model.

&bull; Adjust all the required paths in the `machine_translation/decode.sh` to match those in `machine_translation/train.sh`, and run `machine_translation/decode.sh` to inference your pretrained MT model.

&bull; Use `machine_translation/load_embedding.py` to fetch necessary word embeddings from pretrianed MT model.


### Step 2. Multi-Task Fine-tuning

&bull; Download the [`Hubert-base`](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) pretrained Model without fune-tuning.

&bull; Prepare the MuST-C ST training data, please follow [here](https://github.com/facebookresearch/fairseq/blob/v0.12.2/examples/speech_to_text/docs/mustc_example.md#data-preparation).

&bull; Modify the necessary paths in the `pretrain-all/conf/train_soft_alignment.yaml`, such as:
```
w2v-path=/your/path/to/hubert
mt-model-path=/your/path/to/mt/pretrain/model
decoder-embed-path=/your/path/to/mt/word/embedding
```
&bull; Set data path and other required paths in the `pretrain-all/joint_train_merge.sh`, and run `pretrain-all/joint_train_merge.sh` to fune-tune your model.

&bull; Use `pretrain-all/decode.sh` to inference your model

## Citation