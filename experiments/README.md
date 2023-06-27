# Experiments 

- [segformer-b5-autoencoder-all-data](segformer-b5-autoencoder-all-data): b5 segformer with input 512x512 images and autoencoder trained on all data, even unlabelled.


- [mvp-segformer-norm2](mvp-segformer-norm2): same as `mvp-segformer` but with taken normalization
```math
x \rightarrow (x - \mu) / \sigma,
```
where `\mu` and `\sigma` are per-channel mean and standard deviation of the image.


- [mvp-segformer](mvp-segformer): b5 segformer with input 512x512 images.

