# Uncertaincy estimation with drop out

In real world application cases it is very difficult to have clean data set without noises, for this reason it is essential to have an approach that would be able to estimate the uncertainty of the classification. One of the solution of this problemm may be to considered Bayesian Convolutional Neural Networks that allow to find posterior distribution from which can be derive the uncertainty of the model parameters. However it is computationally intractable since require to perform integration over entire space of parameters of the model[Kwon et al., 2018]. In the recient work by Yarin Gal et al. have been developed a robust method that use the drop out to approximate the Bayesian inference[Gal, 2015].

## Uncertainty analisis

Since the noise data is labeled in the same way as the samples of interests, it is required an approach to to deal with such miss-classifications. In order to over come this problemm one could to evaluate the uncertainty of all test sample and set a threshold that would cut off most of the noise samples.

### Epistemic uncertainty

Using the method proposed in [Kwon et al., 2018] it is possible to estimate aleatoric and epistemic uncertainty. Aleatoric uncertainty capturing noise linked with the observations and epistemic uncertainty accounts for model uncertainty, so it may be used to evaluate the uncertainty on the prediction done by the model. Indeed due to implementing drop out in the evaluation phase, prediction probability will be different in each evaluation experiment and effect of drop out may be seen as randomization of parameters of the model according to the variational predictive distribution[Kwon et al., 2018]

In ordder to calculate the epistemic uncertainty I used formula 4) from [Kwon et al., 2018]:

$$
\begin{align*}
Var(y) &= \frac{1}{T} \sum_{t=1}^T (\hat{p}_t - \bar{p})(\hat{p}_t - \bar{p})^T\\
\bar{p} &= \frac{1}{T} \sum_{t=1}^T \hat{p}\\
\hat{p}_t &= p(\omega_t) = SoftMax(f^{\hat{\omega}_t}(x*))
\end{align*}
$$

,where $f^{\hat{\omega}_t}(x*)$ is the output of the last dense layer.

The implementation of calculus of epistemic uncertainty have been done in the function "get_epistemic_uncertainty", that have been adapted from the code proposed by [Kwon et al., 2018]. Using the mean prediction probability the samples are classified considering label with probability larger than 0.5 (unambiguous classification) the other samples are rejected.

In order to find the distribution of the uncertainty over all samples I run the loop over entire test set and calculated it for each sample. After that I sorted classified samples according to the real labels into corresponding arrays. In this way the noise analysis can be done more effectively and can be avoid the problem of false positive. From the other hand to estimate miss classification of the labels of interest (e.i. "AC", "AD", "H") I store classified labels into dictionary with corresponding real class as keys.

## Model

In order to implement Bayesian Convolutional Neural Network for multi class classification I implement the sequential model composed of 3 convolutional layers each of which is followed by max pooling and dropout layers. As final stage I added the flatten layer that prepare the input for dense layer followed again by drop out. At each new convolutional layer I double the number of filters - starting from 32 - and set the kernel dimension to (3,3), except the first one where I used kernel dimension (5,5). For each layer I used "same" padding and the relu-activation function, therfore I used he_normal-initializer for weight initialization. Instead for the last output dense layer I used softmax-activation function as have been suggested by [Kwon et al., 2018].
Also I forced drop out layer to be active in training and test phase. The main role of the dropout is to randomly turn off some of the inputs to the layer. In this way in training phase it prevent oferfitting, indeed in this way neurons doesn't learn exactly the training data set. From other hand in test phase drop out allow some degree of uncertainty in evaluation and prediction of new samples that allow to approximate posterior distribution[Gal, 2015] and thus estimate variations.

I compile the model with categorical cross entropy as a loss function, Adam optimizer and accuracy as metrics. After that I trained the model for 25 epoch, that seems optimal in training and evaluation performance, since training and evaluation loss/accuracy started to deviate.

Finally I saved the model and history.

### Performance

As may be seen from plot below accuracy and the loss of the train and validation data set present the same behavior, except some fluctuation in validation accuracy and loss. Also both characteristics achieve almost stable and satisfactory values of accuracy and loss. In this way at first analysis the model learn well from training data set and avoid overfitting.

## Evaluate model on test data set

At the next step I evaluated model on the trained data set. As results obtained values of loss is quite hight and accuracy is rather low. This can be explained by the fact that in train and test data present noise that give low contribution to overall accuracy and hight contribution to the loss.

## Prediction rate

From the table below can be seen the number of unambiguously classified sample for each label. Neural Network is able pretty well to recognize classes of interest.

From the other hand noise classes are recognized very bad and most of them are rejected: for example NN wasn't able to discriminate none of samples of glass-class. The rest of noise classes still present very low classification rate. This is desired result since we are are interested into reducing the noise into labeling of the samples of interest.

## Threshold estimation  

At the next step I plotted the density of recognized samples vs corresponding uncertainty. The number of samples have been normalized to the number of recognized samples, in this way the density of different classes can be compared more effectively. from the plot below can be seen the the density of valid samples is highly picked around zero uncertainty.

Instead for the noise samples distribution may be seen that most of the samples have large enough uncertainty and most of them may be cut be threshold. However some of the noise classes present a small pick at zero uncertainty, even if very small w.r.t the valid samples density. Due to this fact some fraction of noise persist among right recognized class of interest. However considering that it's only a fraction of already reduced data set overall number of miss classification labels should be rather low.

# Reference

[Kwon et al., 2018] Kwon, Yongchan, et al. "Uncertainty quantification using bayesian neural networks in classification: Application to ischemic stroke lesion segmentation." (2018).

[Gal, 2015] Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning."(2015)
