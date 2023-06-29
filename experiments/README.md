# Experiments 


- [mvp-segformer-norm2-added-weights](mvp-segformer-norm2-added-weights): same as `mvp-segformer-norm2`, but with weighted loss function. Smaller boxes are weighted more, as ~1/area. 

- [mvp-segformer-norm2-dice-loss](mvp-segformer-norm2-dice-loss): same as `mvp-segformer-norm2` but with dice loss.

- [mvp-segformer-norm2-focal-loss](mvp-segformer-norm2-focal-loss): same as `mvp-segformer-norm2` but with focal loss.

- [segformer-b5-autoencoder-all-data](segformer-b5-autoencoder-all-data): b5 segformer with input 512x512 images and autoencoder trained on all data, even unlabelled.


- [mvp-segformer-norm2](mvp-segformer-norm2): same as `mvp-segformer` but with taken normalization
```math
x \rightarrow (x - \mu) / \sigma,
```
where `\mu` and `\sigma` are per-channel mean and standard deviation of the image.


- [mvp-segformer](mvp-segformer): b5 segformer with input 512x512 images.

