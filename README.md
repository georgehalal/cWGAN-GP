# Conditional Wasserstein Generative Adversarial Network with Gradient Penalty (cWGAN-GP)
## Introduction
A considerable amount of data from galaxy surveys collected by telescopes is thrown away due to subpar observing conditions such as blurriness and noise levels to achieve statistical measurements of high accuracy. Therefore, [Balrog](https://academic.oup.com/mnras/article/457/1/786/988891), a Python-based simulation package, has been developed for minimizing the amount of discarded data. It injects fake simulated galaxies, created by combining noise with real galaxies taken from deep-field surveys, into real astronomical images from wide-field surveys to accurately characterize measurement biases. This is crucial for the final Dark Energy Survey (DES) and first Legacy Survey of Space and Time (LSST) analyses.

However, we can only run the full Balrog pipeline on a few realizations of simulated galaxies in our images, as Balrog is slow and requires very high computational power to run. Therefore, we develop a deep learning model consisting of a conditional Wasserstein Generative Adversarial Network with gradient penalty (cWGAN-GP) that emulates Balrog’s functionality for speeding up the process. A [Wasserstein Generative Adversarial Network (WGAN)](https://ui.adsabs.harvard.edu/abs/2017arXiv170107875A/abstract) is an extension of Generative Adversarial Networks (GANs) that improves the stability of the model when training. However, WGANs can sometimes yield unsatisfactory results or fail to converge. Therefore, by replacing weight clipping in WGANs with penalizing the norm of the gradient of the discriminator with respect to its inputs, a [Wasserstein Generative Adversarial Network with gradient penalty (WGAN-GP)](https://papers.nips.cc/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html) improves the performance of the model. Futhermore, a [conditional GAN](https://arxiv.org/abs/1411.1784) is an extension of a GAN, where additional input layers are added to both the generator and discriminator, allowing a targeted generation of an output of a specific type that depends on the conditions given in the additional input layers.

Since each location in the sky is surveyed multiple times by telescopes producing a stack of images of the same point, different statistics are developed to characterize these observations as a function of position in the sky, including the fluctuation of sky brightness and the blurriness of point-like sources at each point for different passbands (filters). The passbands we consider are r, i, and z, from bluer to redder. Therefore, for each passband and location in the sky, we use two of these observing conditions from the sky maps and the true galaxy magnitude, a measure of brightness, from deep-field surveys as the input in the additional conditioning layers of the cWGAN-GP. We train it on data from Balrog to generate observed galaxy magnitudes in wide-field surveys for the same passbands at the same location in the sky. cWGAN-GPs are useful in this case because for a given set of inputs, the outputs are not deterministic and could vary based on an unknown distribution which a generative model can learn.

## Dataset
* Our training dataset consists of ∼1.7 million galaxies.
* We use 10,000 galaxies for validation.

## Setup
```bash
git clone https://github.com/georgehalal/cWGAN-GP.git
cd cWGAN-GP/
pip install -r requirements.txt
```

## Code Layout
* `preprocess.py` - prepare the raw data into a format useful for the model.
* `model/cWGAN-GP.py` - define the model architecture, the loss function, and the evaluation metrics.
* `model/dataloader.py` - specify how the data should be fed into the model.
* `tests/truecondW1/params.json` - an example directory with a JSON file containing the model and training hyperparameters. Similar directories can be created there containing different hyperparameters. These directories also store the logging info and plots corresponding to the given set of hyperparameters.
* `train_and_evaluate.py` - train the model on the training dataset, evaluating it along the way on the validation dataset
* `search_hyperparams.py` - submit a different job for each set of hyperparameters to search for the best ones.
* `generate.py` - use the generator model to generate a distribution of galaxy magnitudes to compare with the true distribution.
* `utils.py` - contains functions for handling hyperparameters, logging information, and storing model states.

## Performance
![perf](https://github.com/georgehalal/cWGAN-GP/blob/main/img/cWGAN-GP_performance.png)
