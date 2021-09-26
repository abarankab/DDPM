## Denoising Diffusion Probabilistic Models
An implementation of Denoising Diffusion Probabilistic Models for class-conditional image generation written in PyTorch. This roughly follows the original code by Ho et. al. Unlike their implementation, however, my model allows for class conditioning through bias in residual blocks. 

## Experiments

## Resources

## Acknowledgements

I used [Phil Wang's implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) as a reference for my work.

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
