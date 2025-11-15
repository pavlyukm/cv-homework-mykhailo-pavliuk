# CIFAR-10 Classification Experiments Report

|          |          |
|----------|----------|
|Start Date|2025-11-13|
|End Date  |2025-11-15|
|Dataset   |CIFAR-10|
|Author    |Mykhailo Pavliuk|

## Motivation

Conduct systematic experiments on CIFAR-10 to compare:
1. Baseline CNN trained from scratch
2. Transfer learning with DINOv3-base (frozen features)
3. Transfer learning with DINOv3-large (frozen features)
4. Fine-tuning strategies with DINOv3-large:
   - Head-only training
   - Partial unfreezing (last 4 layers)

The goal is to understand the trade-offs between model complexity, training time, and accuracy, and to determine the most effective approach for CIFAR-10 classification.

All experiments were ran on google collab pro A100 GPU

---

# [e0001] Baseline CNN

|          |          |
|----------|----------|
|Start Date|2025-11-13|
|End Date  |2025-11-13|
|Dataset   |CIFAR-10|

## Architecture

Simple CNN with 3 convolutional blocks:
- Block 1: 2x Conv(64) + MaxPool
- Block 2: 2x Conv(128) + MaxPool  
- Block 3: 2x Conv(256) + MaxPool
- FC layers: 4096 → 512 → 10

All conv layers use 3x3 kernels, padding=1, with BatchNorm and ReLU.
Dropout (0.5) applied in classifier.

## Hyperparameters

```
Batch size: 128
Epochs: 5
Learning rate: 0.001
Optimizer: Adam
Data augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip
```

## Results & Deliverables

**Training time**: 30 seconds per epoch
**Best validation accuracy**: 72.52%
**Final train accuracy**: 72.62%
**Total parameters**: 3,249,994
**Trainable parameters**: 3,249,994

## Training Curves

![Loss vs Epoch](e0001_baseline_cnn_metrics.png)

## Interpretation & Conclusion

Just a quick baseline CNN to get started: not very good accuracy, but quick to train. I hope to achieve a lot better results with transfer learning for Dino models.

---

# [e0002] DINOv3-base (Frozen Features)

|          |          |
|----------|----------|
|Start Date|2025-11-13|
|End Date  |2025-11-13|
|Dataset   |CIFAR-10|

## Architecture

- **Backbone**: DINOv3-base (ViT-B/16) - FROZEN
  - Pretrained on LVD-1689M dataset
  - Input size: 224×224 (CIFAR-10 upsampled from 32×32)
- **Classifier**: 
  - Linear(768 → 512) + ReLU + Dropout(0.3)
  - Linear(512 → 10)

## Hyperparameters

```
Batch size: 256
Epochs: 5
Learning rate: 0.001
Optimizer: Adam
Preprocessing: Resize to 224, ImageNet normalization
```

## Results & Deliverables

**Training time**: 38 minutes (457s per epoch)
**Best validation accuracy**: 97.99%
**Final train accuracy**: 98.68%
**Total parameters**: 86,059,274 (398,858 trainable)

## Training Curves

![Loss vs Epoch](e0002_dinov3_base_frozen_metrics.png)

## Interpretation & Conclusion

We can see that pre-trained base model already performs very good on this dataset, and I expected bigger gap between training and vlidation, to be honest. I can also see that the model plateaus almosti mmediately and signs of overfitting. Maybe freezing some features can give us better result, and maybe try large model.

---

# [e0003] DINOv3-large (Frozen Features)

|          |          |
|----------|----------|
|Start Date|2025-11-14|
|End Date  |2025-11-14|
|Dataset   |CIFAR-10|
|Continues |e0002|

## Architecture

- **Backbone**: DINOv3-large (ViT-L/16) - FROZEN
  - Larger model than base
  - Input size: 224×224
- **Classifier**: Same as e0002
  - Linear(1024 → 512) + ReLU + Dropout(0.3)
  - Linear(512 → 10)

## Hyperparameters

```
Batch size: 256
Epochs: 5
Learning rate: 0.001
Optimizer: Adam
```

## Results & Deliverables

**Training time**: 104 minutes (1256s per epoch)
**Best validation accuracy**: 99.08%
**Final train accuracy**: 99.52%
**Total parameters**: 303,659,530 (529,930 trainable)

## Training Curves

![Loss vs Epoch](e0003_dinov3_large_frozen_metrics.png)

## Interpretation & Conclusion

Larger model improved accuracy, and I think it's worth the computational and time trade off. It also shows that it is overfitting and it converged immediately, but I don't think ti can get better than this accuracy. I will still ttry to fine tune it bu running training with all layers except head frozen and also once partially unfreezing.

---

# [e0004a-c] DINOv3-large Fine-tuning

|          |          |
|----------|----------|
|Start Date|2025-11-14|
|End Date  |2025-11-14|
|Dataset   |CIFAR-10|
|Continues |e0003|

## Experiments Overview

Two fine-tuning strategies:
- **e0004a**: Train classifier head only (warmup)
- **e0004b**: Unfreeze last 4 transformer layers

---

## [e0004a] Head-Only Fine-tuning

### Configuration
```
Backbone: FROZEN
Classifier: TRAINABLE
Epochs: 5
Learning rate: 0.001
```

### Results
**Best validation accuracy**: 98.97%
**Training time**: 104 minutes

![Loss vs Epoch](e0004a_dinov3_large_head_metrics.png)

---

## [e0004b] Partial Fine-tuning (Last 4 Layers)

### Configuration
```
Backbone: Last 4 layers UNFROZEN
Classifier: TRAINABLE
Epochs: 5
Learning rate: 0.0001 (reduced for stability)
Starting from: e0004a checkpoint
```

### Results
**Best validation accuracy**: __.___%
**Training time**: ___ minutes

![Loss vs Epoch](e0004b_dinov3_large_partial_metrics.png)


## Fine-tuning Strategy Comparison

| Experiment |     Strategy    | Val Acc | Training Time | Trainable Params |
|------------|-----------------|---------|---------------|------------------|
| e0004a     | Head only       | 98.97% | 104 min       | 529,930         | 
| e0004b     | Last 4 layers   | 99.32% | 125 min       | 50,918,922         |

## Interpretation & Conclusion

[FILL IN AFTER RUNNING]
- Which strategy achieved best accuracy?
- Trade-off between accuracy and training time
- Signs of overfitting in full fine-tuning?
- Optimal learning rate for each strategy

---

# Overall Comparison

## Summary Table

| Experiment | Description | Val Acc | Train Time | Total Params | Trainable Params |
|------------|-------------|---------|------------|--------------|------------------|
| e0001      | Baseline CNN | __.__% | ___ min   | ___,___     | ___,___         |
| e0002      | DINOv3-base (frozen) | __.__% | ___ min | ___,___ | ___,___ |
| e0003      | DINOv3-large (frozen) | __.__% | ___ min | ___,___ | ___,___ |
| e0004a     | DINOv3-large (head) | __.__% | ___ min | ___,___ | ___,___ |
| e0004b     | DINOv3-large (partial) | __.__% | ___ min | ___,___ | ___,___ |

## Comparison Charts

![All Experiments Comparison](experiments_comparison.png)

## Key Findings

[FILL IN AFTER RUNNING]

### Best Overall Model
**Winner**: _______
**Accuracy**: __.___%
**Justification**: _______

### Efficiency vs Accuracy
- Most efficient (accuracy/time): _______
- Best accuracy regardless of cost: _______
- Best for production deployment: _______

### Transfer Learning Insights
- Baseline CNN performance: __.___%
- Best frozen features model: _______
- Improvement from fine-tuning: +___._%

### Recommendations
1. For quick prototyping: _______
2. For best accuracy: _______
3. For production deployment: _______

## Conclusions

[FILL IN AFTER RUNNING]

**Main takeaways**:
1. _______
2. _______
3. _______

**Future work**:
- Try DINOv3-small for faster training and do a full fine-tune
- Increase drop-out
- Try different augmentation strategies

---

