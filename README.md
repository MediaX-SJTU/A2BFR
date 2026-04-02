<div align="center">
<h2>A<sup>2</sup>BFR: Attribute-Aware Blind Face Restoration</h2>

<a href='https://arxiv.org/abs/2603.29423'>

</div>


## ⏰ Update
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

Coming soon...

## 🍭 Inference with script

Coming soon...

## 🔥 Training

Coming soon...


## License
This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
This project is based on [OminiControl](https://github.com/Yuanshi9815/OminiControl).
We also leveraged [facer](https://github.com/FacePerceiver/facer)'s code in our project.
Thanks for these awesome work!