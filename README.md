# A-ReSEUnet

## 1. **Repository Structure**

### Datasets/

- **CPM17/** - From [HoVerNet](https://github.com/vqdang/hover_net)
- **CoNSeP/** - From [HoVerNet](https://github.com/vqdang/hover_net)
- **MoNuSeg/** - From [MoNuSeg](https://monuseg.grand-challenge.org/Data/) (Labels have been converted to .mat)
- **crop_patches.py** - Crop to specified size

### MasterPaper_class/

Suitable for multiclass segmentation tasks

### MasterPaper_new/

Suitable for binary segmentation tasks

### SimCLR pretrained/(Add it yourself)

Download the pretrained weight provided by the paper

**[self-supervised-histopathology](https://github.com/ozanciga/self-supervised-histopathology)**

## 2. Running the Code

### Training

- **config.yaml** - Contains many hyperparameters and all folder path details
- **main.py** - Main training script

```python
python main.py
```

### Inference

- **predict.py** - Main inference script
- **post.py** - Main postprocessing script
- **compute_stats.py** - Compute metrics (Dice, AJI, DQ, SQ, PQ, AJI+)