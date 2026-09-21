# Experiment 9
### Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

## Aim
Compare a one-vs-rest perceptron (PLA) written from scratch in numpy with a PyTorch MLP on 62-class handwritten character images (3,410 images, cropped and resized to 32x32), tune the MLP's activation, cost function, optimizer, learning rate, batch size, depth and width on a validation split, and study convergence and overfitting.

## Final MLP Hyperparameters

| Hyperparameter | Chosen Value |
|---|---|
| Architecture | 1 hidden layer, 512 neurons |
| Activation | sigmoid |
| Cost function | cross-entropy |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 512 |
| Overfitting remedy | random rotation, scaling and shift augmentation, 300 epochs (no dropout, weight decay or early stopping) |

## Test Set Performance (mean of 5 seeds)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| PLA | 0.6930 +/- 0.0172 | 0.7240 +/- 0.0114 | 0.6923 +/- 0.0171 | 0.6931 +/- 0.0174 |
| Softmax, no hidden layer | 0.7195 +/- 0.0052 | 0.7309 +/- 0.0058 | 0.7198 +/- 0.0051 | 0.7158 +/- 0.0053 |
| MLP without augmentation | 0.7785 +/- 0.0034 | 0.7892 +/- 0.0025 | 0.7780 +/- 0.0033 | 0.7752 +/- 0.0032 |
| Tuned MLP (with augmentation) | **0.8305 +/- 0.0053** | **0.8409 +/- 0.0047** | **0.8301 +/- 0.0051** | **0.8273 +/- 0.0045** |

ROC AUC on the test set (seed 42): PLA micro 0.9565, macro 0.9558; tuned MLP micro 0.996, macro 0.9961.

## Test Accuracy by Character Type (seed 42)

| Model | Overall | Digits | Uppercase | Lowercase | Upper/lower case counted as same |
|---|---|---|---|---|---|
| PLA | 0.7148 | 0.7654 | 0.7870 | 0.6233 | 0.7656 |
| MLP without augmentation | 0.7832 | 0.8148 | 0.8565 | 0.6977 | 0.8438 |
| Tuned MLP | 0.8379 | 0.8148 | 0.9167 | 0.7674 | 0.8926 |

## Hyperparameter Impact (180-run grid, validation accuracy)

| Hyperparameter | Spread of mean val. acc. across values | Spread of best run across values |
|---|---|---|
| optimizer | 0.421 | 0.146 |
| cost | 0.261 | 0.014 |
| lr | 0.114 | 0.006 |
| batch | 0.113 | 0.006 |
| activation | 0.077 | 0.014 |

Changing one hyperparameter at a time away from the best grid run (validation accuracy points): GD -73.8, plain SGD -52.9, SGD+momentum -15.6, learning rate 0.1 -18.9, batch 32 -8.8, MSE cost -1.4, ReLU -2.3, tanh -1.4. The naive setup (sigmoid, MSE, plain SGD, lr 0.01, batch 32) reached 6.8% validation accuracy.

## Optimizer Convergence (sigmoid, cross-entropy, 100 epochs)

| Optimizer | Settings tried | Best val. acc. (%) | Settings reaching 90% train acc. | Epochs to 90% train acc. at its best setting |
|---|---|---|---|---|
| GD | 3 | 37.3 | 0 / 3 | never |
| SGD | 9 | 76.0 | 1 / 9 | 64 |
| SGD+momentum | 9 | 77.0 | 4 / 9 | 11 |
| Adam | 9 | 77.7 | 6 / 9 | 12 |

## Number of Hidden Layers (width 256, mean of 4 seeds)

| Hidden layers | Best-grid settings: train acc. (%) | Best-grid settings: val. acc. (%) | Adam 0.001, ReLU: val. acc. (%) | tanh (%) | sigmoid (%) |
|---|---|---|---|---|---|
| 0 | 99.7 | 72.8 | - | - | - |
| 1 | 99.9 | 77.8 | 75.5 | 76.8 | 77.5 |
| 2 | 99.9 | 77.0 | 77.0 | 76.8 | 76.5 |
| 3 | 81.9 | 58.7 | 75.0 | 77.3 | 73.0 |
| 4 | 9.2 | 6.7 | 74.2 | 76.3 | 61.6 |
| 5 | 2.4 | 2.2 | 73.0 | 74.9 | 51.3 |

## Overfitting Remedies (chosen architecture, mean of 4 seeds)

| Setting | Train acc. (%) | Val. acc. (%) | Gap (points) |
|---|---|---|---|
| no mitigation | 99.9 | 77.9 | 22.0 |
| dropout 0.4 | 99.7 | 77.7 | 21.9 |
| weight decay 1e-3 | 93.9 | 75.7 | 18.2 |
| dropout 0.4 + weight decay 1e-3 | 92.5 | 77.1 | 15.4 |
| early stopping (patience 10) | 96.0 | 77.6 | 18.4 |
| augmentation, 100 epochs | 96.0 | 82.0 | 14.0 |
| augmentation, 300 epochs | 97.3 | **82.8** | 14.6 |
| augmentation + dropout 0.4 + weight decay 1e-3, 300 epochs | 86.2 | 78.7 | 7.5 |

## Learning Outcomes
- The perceptron fits the training images (99.2% accuracy) but reaches only 69.3% on the test set. A softmax classifier with the same weights gets 72.0%, a hidden layer adds 5.9 points (77.9%), and augmentation, tried only with the MLP, adds 5.2 more (83.0%).
- The optimizer and its learning rate mattered most and the activation least: GD, plain SGD and momentum SGD lose 15.6 to 73.8 points against Adam at the best setting, while switching activation costs at most 2.3. The untuned sigmoid + MSE + SGD setup reached only 6.8%.
- Adam reached 90% training accuracy in 12 epochs against 64 for plain SGD, and 6 of 9 Adam settings got there against 1 of 9 for SGD; the best validation accuracies were close (76.0% for SGD, 77.7% for Adam).
- More hidden layers did not help: one layer gave 77.8% and two gave 77.0%, sigmoid at learning rate 0.01 failed to train from four layers on, and sigmoid with Adam 0.001 fell to 51.3% at five layers.
- The MLP overfits (99.9% train against 77.9% validation). Only augmentation raised validation accuracy (82.8%); dropout and early stopping gave no gain and weight decay shrank the gap but lowered accuracy. About a third of the remaining test errors (28 of 83) are a letter predicted as its other-case twin (v and V, w and W), and the most frequent single confusion is 0 predicted as O.
