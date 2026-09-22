# Experiment 9
### Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

## Aim
Compare a one-vs-rest perceptron (PLA) written from scratch in numpy with a tuned PyTorch MLP on 62-class handwritten character images (3,410 images, cropped and resized to 32x32), tuning activation, cost function, optimizer, learning rate, batch size, depth and width on a validation split.

## Final MLP
1 hidden layer (512 sigmoid units), cross-entropy loss, Adam, lr=0.01, batch=512, with data augmentation for 300 epochs.

## Test Set Performance (mean of 5 seeds)

| Model | Accuracy | F1 (macro) | ROC AUC (micro) |
|---|---|---|---|
| PLA | 0.6930 ± 0.0172 | 0.6931 | 0.9565 |
| MLP, no augmentation | 0.7785 ± 0.0034 | 0.7752 | -- |
| Tuned MLP | **0.8305 ± 0.0053** | **0.8273** | **0.9960** |

## Learning Outcomes
- Learned that fitting training data isn't the same as generalizing: the perceptron memorized 99.2% of training images but only classified 69.3% of test images correctly.
- Learned that the optimizer decides whether training works at all, more than the activation function does: switching to Adam gained up to 73.8 points, switching activation gained at most 2.3.
- Learned to check training curves, not just final numbers: Adam reached 90% training accuracy in 12 epochs against 64 for plain SGD, though both ended close on accuracy.
- Learned that a deeper network isn't automatically better: accuracy fell sharply past two or three hidden layers, and the deepest sigmoid networks stopped training altogether.
- Learned that not every regularizer fixes overfitting: dropout, weight decay and early stopping left the train-validation gap almost unchanged, and only data augmentation actually closed it.
