<div align="center">
<h2>A<sup>2</sup>BFR: Attribute-Aware Blind Face Restoration</h2>

<a href='https://arxiv.org/abs/2603.29423'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/thetrigger/A2BFR'><img src='https://img.shields.io/badge/Model-HuggingFace-yellow'></a>

</div>


## ⏰ Update
- **2026.06.24**: Inference code and model weights are released.
- **2025.12.30**: Repo is created.


:star: If A<sup>2</sup>BFR is helpful to you, please help star this repo. Thanks! 

## 🌟 Overview
<div align="center">
  <img src="static/images/teaser.png" alt="" width="80%">
</div>

1. Built on **FLUX.1-dev**, we train an **Attribute-Aware** blind face restoration (BFR) model using **LoRA**, and inject the low-resolution (LR) condition via **sequence concatenation**.  
2. With our **Attribute-Aware Learning** and **Semantic Dual Training** strategy, the BFR process can be jointly guided by text prompts, enabling **strong and accurate control over facial attributes**.  
3. We construct a large-scale face editing dataset **AttrFace-90K** primarily based on **FFHQ** and **ReFaceHQ**, **doubling** the dataset scale and the number of supported attributes compared to existing open-source datasets.



## 😍 Visual Results
### Customizable Results with Attribute Prompts

<div align="center">
  <img src="static/images/visual.png" alt="" width="80%">
</div>

**A<sup>2</sup>BFR** enables customizable restoration via attribute prompts. Each column presents diverse outputs generated from the same low-quality input under different facial attribute conditions (e.g., *smiling*, *eyeglasses*), demonstrating controllable, user-directed face restoration. 

<div align="center">
  <img src="static/images/ood.png" alt="" width="80%">
</div>
**A<sup>2</sup>BFR** exhibits strong generalization beyond the 12 attribute categories defined in AttrFace-90K. It can respond to out-of-domain attribute prompts and produce the corresponding edits, demonstrating robust cross-attribute generalization.

### Comparison with BFR Methods

<div align="center">
  <img src="static/images/quality_visual.png" alt="" width="80%">
</div>

**A<sup>2</sup>BFR** delivers superior restoration quality compared with existing BFR methods.

### Comparison with Restore-then-Edit Pipelines

<div align="center">
  <img src="static/images/visual2.png" alt="" width="80%">
</div>

Restore-then-edit pipelines often compromise fidelity to the low-quality (LQ) input. In contrast, **A<sup>2</sup>BFR** produces restorations that are both **faithful** to the input and **aligned** with the specified attributes.

## 🗂️ Dataset
<div align="center">
  <img src="static/images/AttrFace.png" alt="" width="90%">
</div>

**Left:** Examples of image and prompt pairs in AttrFace-90K.  
**Right:** A quick view of AttrFace-90K.



## ⚙ Dependencies and Installation

```bash
conda create -n a2bfr python=3.10 -y
conda activate a2bfr
pip install -r requirements.txt
```

Please install a CUDA-compatible PyTorch build for your GPU environment if it is not already available.

## 🍭 Inference with script

This repository currently releases inference code only. Evaluation scripts, datasets, private experiment paths, and training code are not included in this release.

You need two model paths:

- `--base-model`: the FLUX.1-dev base model directory or Hugging Face model id.
- `--lora`: the A<sup>2</sup>BFR LoRA checkpoint, usually `default.safetensors`.

Download the released A<sup>2</sup>BFR LoRA checkpoint from Hugging Face:

```bash
hf download thetrigger/A2BFR default.safetensors --local-dir checkpoints/A2BFR
```

The model weights are hosted at [thetrigger/A2BFR](https://huggingface.co/thetrigger/A2BFR). Please download or prepare `FLUX.1-dev` separately and follow its license/usage terms.

Single image inference:

```bash
python infer.py \
  --base-model /path/to/FLUX.1-dev \
  --lora checkpoints/A2BFR/default.safetensors \
  --condition-image /path/to/lq_face.png \
  --output-dir outputs/demo \
  --prompt "A photo of a human face" \
  --guidance-scale 3.0 \
  --seed 0 \
  --concat
```

Batch inference:

```bash
python infer.py \
  --base-model /path/to/FLUX.1-dev \
  --lora checkpoints/A2BFR/default.safetensors \
  --input-dir /path/to/lq_faces \
  --output-dir outputs/batch \
  --prompt "A high-quality, high-resolution, realistic, and extremely detailed image of a human face" \
  --guidance-scale 3.0 \
  --seed 0
```

Useful options:

- `--prompt`: controls facial attributes with text.
- `--width` / `--height`: default to `512`.
- `--dtype`: defaults to `bfloat16`; use `float16` if needed.
- `--preprocess deblurring`: creates a degraded condition image from a clean input. For already low-quality inputs, keep the default `--preprocess none`.
- `--save-condition`: saves the resized/preprocessed condition image.

## 🔥 Training

Training code will be released.


## License
This project is released under the [Apache 2.0 license](LICENSE.txt).

## Acknowledgement
This project is based on [OminiControl](https://github.com/Yuanshi9815/OminiControl).
We also leveraged [facer](https://github.com/FacePerceiver/facer)'s code in our project.
Thanks for these awesome work!