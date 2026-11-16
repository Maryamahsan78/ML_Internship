# ML Internship, Deep Learning Phase 2: Convolutional Neural Networks (CNNs)

Six notebooks that build up CNNs for image classification: convolution by hand, CNNs in PyTorch, a Fashion-MNIST project, CIFAR-10 with data augmentation, ResNet and residual connections, and transfer learning with a pretrained ResNet-18.

**Author:** Maryam ([GitHub: Maryamahsan78](https://github.com/Maryamahsan78))

## Notebooks

|File|Topic|Data|
|-|-|-|
|`convolution\\\\\\\_from\\\\\\\_scratch.ipynb`|Images as numbers, 2D convolution in NumPy, filters and feature maps, padding, stride, output size, ReLU, max pooling, learnable filters|Sample photo, digits|
|`pytorch\\\\\\\_fashion\\\\\\\_mnist.ipynb`|`nn.Conv2d`, `nn.MaxPool2d`, shape tracing, training on the GPU, CNN vs MLP|Fashion-MNIST|
|`fashion\\\\\\\_project.ipynb`|Batch normalization, dropout, final test evaluation, confusion matrix, misclassified images, learned filters and feature maps|Fashion-MNIST|
|`cifar10\\\\\\\_augmentation.ipynb`|Colour images, data augmentation, learning-rate scheduler, overfitting gap|CIFAR-10|
|`architectures\\\\\\\_resnet.ipynb`|LeNet, VGG, ResNet, residual block built by hand, ResNet-20 vs the same network without skip connections|CIFAR-10|
|`transfer\\\\\\\_learning.ipynb`|Pretrained ResNet-18, feature extraction vs fine-tuning vs training from scratch, predicting a single image|CIFAR-10 subset (5,000 images)|

## What I learned

* **Convolution:** a small filter slides over the image and produces a feature map. Output size = `floor((n + 2p - f) / s) + 1`. A conv layer shares its filter across the image, so it needs far fewer parameters than a dense layer (160 vs about 9.8 million in my example).
* **Filters are learned:** starting from a random 3x3 filter, gradient descent found the Sobel edge filter by itself.
* **CNN building blocks:** Conv -> BatchNorm -> ReLU -> MaxPool, then dropout and dense layers. I traced the tensor shape through every layer.
* **Regularization for images:** batch normalization, dropout and data augmentation (applied to the training images only).
* **ResNet:** a residual block computes `F(x) + x`, which lets very deep networks train.
* **Transfer learning:** reuse ImageNet-pretrained layers, replace the last layer, and use a smaller learning rate for pretrained layers.

## Results

|Experiment|Result|
|-|-|
|MLP on Fashion-MNIST (Day 2)|validation accuracy 0.889, 235,146 parameters|
|Simple CNN on Fashion-MNIST (Day 2)|validation accuracy 0.918, 105,866 parameters|
|CNN with batch norm and dropout (Day 3)|validation accuracy 0.922 (plain CNN: 0.911), **test accuracy 0.916**|
|CIFAR-10 CNN, no augmentation / augmentation / augmentation + cosine LR (Day 4)|validation accuracy 82.92 / 85.94 / 88.28 ; test accuracy of the chosen model87.02|
|ResNet-20 vs the same network without skip connections (Day 5)|validation accuracy 0.8418 vs 0.8206 (both 272,474 parameters)|
|ResNet-18 on 5,000 CIFAR-10 images: scratch / head only / fine-tuning (Day 6)|validation accuracy 0.494 / 0.814 / 0.864 ; test accuracy of the chosen model 0.8697|

### Explanation

On Fashion-MNIST the CNN beat the MLP (0.918 vs 0.889 validation accuracy) with less than half the parameters, because convolutions use the 2D structure of the image. Batch normalization and dropout gave a further small gain (0.922) and the final test accuracy was 0.916. The hardest class was shirt (F1 0.748), mostly confused with T-shirt, followed by coat vs pullover, which look alike at 28x28 pixels.

On CIFAR-10, data augmentation improved validation accuracy from 82.92% to 85.94%, and adding a cosine learning rate scheduler increased it further to 88.28%. The exact reduction in the training and validation loss gap cannot be determined because the loss values were not recorded. ResNet-20 with skip connections achieved 84.18% validation accuracy, compared with 82.06% without skip connections, showing a small improvement of 2.12 percentage points. Pretrained ResNet-18 with fine-tuning achieved 86.4% validation accuracy, while training from scratch achieved only 49.4%, a significant improvement of 37 percentage points. Small differences should be interpreted carefully because they may be affected by validation-set noise.



Tools

Python, NumPy, PyTorch, torchvision, scikit-learn, pandas, matplotlib, seaborn, Google Colab (T4 GPU)

## How to run

1. Open a notebook in [Google Colab](https://colab.research.google.com) (File > Upload notebook, or open it from GitHub).
2. For Days 2 to 6, switch on the GPU: Runtime > Change runtime type > T4 GPU. Then click **Connect**.
3. The datasets (Fashion-MNIST, CIFAR-10) and the pretrained ResNet-18 weights download automatically.
4. Run **Runtime > Run all**. Each notebook takes about 5 to 15 minutes on the GPU.

