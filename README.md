# Transfer Learning with MobileNetV2
Using the neural network MobileNetV2 to classify Google's image dataset "cats and dogs".
 
## Details
The "cats and dogs" dataset, used in this transfer learning showcase, is composed of thousands of images of cats and dogs. Since this is a transfer learning project, the MobileNet used was not trained in "cats and dogs", but rather on the ImageNet dataset. 
 
Besides using the weights of the already trained MobileNetV2, in the training script a couple of other layers are added to the neural network so that we achieve the desired results.
 
## References
[MobileNetV2] is a concise convolutional neural network efficient enough to be deployed in mobile devices with very limited computing resources. It was first featured in the paper: Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
 
[ImageNet] is an image dataset consisting in more than 14 million images of objects belonging to more than 20 thousand different classes. First featured in the paper: Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009, June). Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition (pp. 248-255). Ieee.

[ImageNet]: https://www.image-net.org
[MobileNetV2]: https://www.image-net.org](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
