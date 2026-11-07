# ML Internship, Deep Learning Phase 1: Foundations of Neural Networks

Five notebooks that build a neural network from the maths up: a NumPy network from scratch, the same network in PyTorch, a Breast Cancer classifier, and regularization with a multi-class Iris model.

**Author:** Maryam ([GitHub: Maryamahsan78](https://github.com/Maryamahsan78))

## Notebooks

|File|Topic|Data|
|-|-|-|
|`neuron\_forward\_pass.ipynb`|Neuron, weights and bias, activation functions (ReLU, Sigmoid, Tanh), layers and shapes, forward pass, why non-linearity is needed (XOR), softmax|Two moons, XOR|
|`backprop\_from\_scratch.ipynb`|Loss functions, gradient descent, backpropagation, a full MLP in NumPy, gradient check, learning rate|Two moons|
|`pytorch.ipynb`|Tensors, autograd, `nn.Module`, `DataLoader`, training loop, NumPy vs PyTorch, SGD vs momentum vs Adam|Two moons|
|`breast\_cancer\_mlp.ipynb`|MLP classifier project with train/validation/test split, learning curves, comparison with Week 3 models|Breast Cancer|
|`regularization\_iris.ipynb`|Overfitting and underfitting, dropout, weight decay, early stopping, multi-class classifier|Breast Cancer, Iris|

## What I learned

* **Forward pass:** each layer computes `activation(X @ W + b)`. Without non-linear activations, stacked layers collapse into one linear layer.
* **Training:** a loss function measures the error, backpropagation (the chain rule) computes the gradient of every weight, and an optimizer updates the weights. I coded this by hand in NumPy and verified it with a gradient check (relative error about 1e-10).
* **PyTorch:** autograd computes the same gradients automatically. A training loop is five lines: `zero\_grad`, forward, loss, `backward`, `step`.
* **Optimizers:** with the same model and mini-batches, plain SGD needed 66 epochs to reach a loss below 0.4, SGD with momentum 8 and Adam 4.
* **Regularization:** dropout, weight decay and early stopping reduce the gap between training and validation loss.

## Results

|Experiment|Result|
|-|-|
|NumPy MLP on two moons (Day 2)|test accuracy 0.880|
|PyTorch MLP on two moons, full-batch SGD (Day 3)|test accuracy 0.893|
|PyTorch MLP on Breast Cancer (Day 4)|test accuracy 0.965, malignant recall 0.981 (1 of 53 malignant tumours missed)|
|Logistic regression on the same split (Day 4)|test accuracy 0.986|
|Big MLP without regularization (Day 5)|lowest validation loss 0.0067 at epoch 57, rising to 0.0175 by epoch 200 (overfitting)|
|Same model with dropout + weight decay + early stopping (Day 5)|validation loss 0.0051, stopped at epoch 69|
|MLP on Iris, 3 classes (Day 5)|test accuracy 0.933 (30 flowers)|

### Explanation

On the Breast Cancer data the neural network reached 0.965 test accuracy, but it did **not** beat simple models: logistic regression scored 0.986 on the same split. The difference is 3 patients out of 143, which is within the noise of such a small test set, and neural networks usually do not win on small tabular data. The learning curves showed overfitting (training loss near zero while validation loss rose again). Regularization fixed this in the validation loss (0.0175 down to 0.0051) but did not change the test accuracy, which stayed at 0.965, because the dataset is easy and the test set is small. I selected models using the validation set and used the test set once, at the end.

## Key takeaways

* Know what every step of a network computes, not only how to call the library.
* Compare against simple baselines and judge differences against the noise (one test patient is about 0.007 accuracy).
* Choose models on validation data, and use the test set only once.
* Watch the learning curves: the gap between training and validation loss tells you when to regularize.

## Tools

Python, NumPy, PyTorch, scikit-learn, pandas, matplotlib, seaborn, Google Colab

## How to run

1. Open a notebook in [Google Colab](https://colab.research.google.com) (File > Upload notebook, or open it from GitHub) and click **Connect**.
2. All datasets are built into scikit-learn, so nothing needs uploading. PyTorch is already installed in Colab.
3. Run **Runtime > Run all**. No GPU is needed for this phase.

