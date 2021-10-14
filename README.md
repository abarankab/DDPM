## Denoising Diffusion Probabilistic Models

An implementation of Denoising Diffusion Probabilistic Models for image generation written in PyTorch. This roughly follows the original code by Ho et al. Unlike their implementation, however, my model allows for class conditioning through bias in residual blocks. 

## Experiments

I have trained the model on MNIST and CIFAR-10 datasets. The model seemed to converge well on the MNIST dataset, producing realistic samples. However, I am yet to report the same CIFAR-10 quality that Ho. et al. provide in their paper. Here are the samples generated with a linear schedule after 2000 epochs:

![Samples after 2000 epochs](resources/samples_linear_200.png)

Here is a sample of a diffusion sequence on MNIST:

<p align="center">
  <img src="resources/diffusion_sequence_mnist.gif" />
</p>


## Resources

I gave a talk about diffusion models, NCSNs, and their applications in audio generation. The [slides are available here](resources/diffusion_models_talk_slides.pdf).

I also compiled a report with what are, in my opinion, the most crucial findings on the topic of denoising diffusion models. It is also [available in this repository](resources/diffusion_models_report.pdf).


## Acknowledgements

I used [Phil Wang's implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) and [the official Tensorflow repo](https://github.com/hojonathanho/diffusion) as a reference for my work.

## Citations

```bibtex
@misc{ho2020denoising,
    title   = {Denoising Diffusion Probabilistic Models},
    author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year    = {2020},
    eprint  = {2006.11239},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@inproceedings{anonymous2021improved,
    title   = {Improved Denoising Diffusion Probabilistic Models},
    author  = {Anonymous},
    booktitle = {Submitted to International Conference on Learning Representations},
    year    = {2021},
    url     = {https://openreview.net/forum?id=-NEXDKk8gZ},
    note    = {under review}
}
