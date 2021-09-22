# About
f-Cal is calibration method proposed to calibrate probabilistic regression networks. Typical bayesian neural networks are shown to be overconfident in their predictions. To use the predictions for downstream tasks, reliable and *calibrated* uncertainity estimates are critical. f-Cal is a straightforward loss function, which can be employed to train any probabilistic neural regressor, and obstain calibrated uncertainty estimates.

# Abstract
While modern deep neural networks are performant perception modules, performance (accuracy) alone is insufficient, particularly for safety-critical robotic applications such as self-driving vehicles. Robot autonomy stacks also require these otherwise blackbox models to produce reliable and calibrated measures of confidence on their predictions. Existing approaches estimate uncertainty from these neural network perception stacks by modifying network architectures, inference procedure, or loss functions. However, in general, these method slack calibration, meaning that the predictive uncertainties do not faithfully represent the true underlying uncertainties(process noise). Our key insight is that calibration is only achieved by imposing constraints across multiple examples, such as those in a mini-batch; as opposed to existing approaches which only impose constraints per-sample, often leading to overconfident (thus miscalibrated) uncertainty estimates. By enforcing the distribution of outputs of a neural network to resemble a target distribution by minimizing an f-divergence, we obtain significantly better-calibrated models compared to prior approaches. Our approach, f-Cal, outperforms existing uncertainty calibration approaches on robot perception tasks such as object detection and monocular depth estimation over multiple real-world benchmarks.

# Pipeline

![alt text](./figures/pipeline-ICRA-dhaivat.png)

**f-Cal:** We make a conceptually simple tweak to the loss function in a typical (deterministic) neural network training pipeline. In addition to the empirical risk (e.g.,L1, L2, etc.) terms, we impose a distribution matching constraint ($L_{f−Cal}$) over the error residuals  across  a  mini-batch.  By  encouraging  the  distribution  of  these  error  residuals  to  match  a  target calibrating  distribution(e.g.,Gaussian),  we  ensure  the  neural  network  predictions  are calibrated.  Compared  to  prior  approaches, most  of  which  perform  post-hoc calibration, or require large held-out calibration datasets, f-Cal does not impose an inference time overhead. f-Cal is task and architecture agnostic, and we apply it to robot perception problems such as object detection and depth estimation.

# Algorithm

f-Cal is a conceptually simple algorithm which can be implemented into any standard autodifferentiation tools such as [pytorch](https://pytorch.org) or [tensorflow](https://tensorflow.org). Through probabilistic neural regressor, we get a set of predictions($\phi_i = (\mu_i, \sigma_i$)) for groundtruth $y_i$.  We construct residuals belonging to standard normal distribution, and construct chi-squared distribution to calculate f-divergence. 

# Quantitative results:

We evaluate f-Cal for a wide range of robot perception tasks and datasets. In each column group (a, b, c, d), we report an empirical risk (deterministic performance metric such as L1, SiLog, RMSE, mAP), expected calibration errors (ECE), and negative log-likelihood. f-Cal consistently outperforms all other calibration techniques considered (lower ECE values). (**Note**: L1 scores are scaled by a factor of 1000 and ECE scores by a factor 100 for improved readability. ↓: Lower is better, ↑: Higher is better, −: Method did not scale to task/dataset) 


# Qualitative results:

## Object detection:
In object detection, we observe that NLL trained models yield substantially overconfident predictions, even for cars which are occluded. While f-Cal yields low uncertainty for 

## Depth estimation: