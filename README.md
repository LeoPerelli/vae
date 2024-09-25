# vae

# How does a Variational Auto Encoder work?
A Variational auto encoder is an encoder-decoder neural network. 
In a VAE, we assume there is some latent variable $\mathbf{z}$ that is not observable but is a compression of $\mathbf{x}$.
Our aim is to learn the data distribution $p(\mathbf{x})$, as well as the conditional distributions $p(\mathbf{x}|\mathbf{z})$ and $p(\mathbf{z}|\mathbf{x})$.
For simplicity, assume that $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and we refer to it as the prior. 
Then, we try to learn some parametrized distribution $p_{\theta}(\mathbf{x} | \mathbf{z})$ which approximates $p(\mathbf{x}|\mathbf{z})$.
In this case, the distribution is parametrized as a neural network.
By Bayes theorem, we know that:
```math
p_{\theta}(\mathbf{x}) = \frac{p_{\theta}(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{p_{\theta}(\mathbf{z} | \mathbf{x})}
```
Since we use a neural network to learn $p_{\theta}(\mathbf{x} | \mathbf{z})$, we cannot invert it easily to obtain $p_{\theta}(\mathbf{z} | \mathbf{x})$.
Hence, we approximate $p_{\theta}(\mathbf{z} | \mathbf{x})$ with another parametrized distribution $q_{\phi}(\mathbf{z} | \mathbf{x})$. 
The VAE thus is an auto encoder where $p_{\theta}$ is the encoder and $q_{\phi}$ is the decoder.
The training objective is to maximize the log likelihood of the dataset. 
By computing the log likelihood, one can derive that: 
```math
log \text{ } p_{\theta}(\mathbf{x} | \mathbf{z} ) = KL(q_{\phi}(\mathbf{z} | \mathbf{x}) || p_{\theta}(\mathbf{z} | \mathbf{x})) + \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})}[ - log \text{ } q_{\phi}(\mathbf{z} | \mathbf{x})  + log \text{ } p_{\theta} (\mathbf{x}, \mathbf{z}) ] \geq \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})}[ - log \text{ } q_{\phi}(\mathbf{z} | \mathbf{x})  + log \text{ } p_{\theta} (\mathbf{x}, \mathbf{z})]
```
Since the KL divergence is greater or equal to 0, the expected value is referred to as lower evidence bound, as it is the lower bound for the value of the "evidence" $log \text{ } p_{\theta}(\mathbf{x} | \mathbf{z} )$.
It is possible to futher derive that the lower bound is equal to:
```math
\mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})}[ - log \text{ } q_{\phi}(\mathbf{z} | \mathbf{x})  + log \text{ } p_{\theta} (\mathbf{x}, \mathbf{z}) ] =  - KL(q_{\phi}(\mathbf{z} | \mathbf{x}) \text{ || } p(\mathbf{z}) ) + \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})}[ log \text{ } p_{\theta}( \mathbf{x} | \mathbf{z}) ]
```  
The two terms have a clear meaning. 
The KL divergence acts as a regularization on the latent space, forcing $\mathbf{z} | \mathbf{x}$ to be close to the prior.
The expectation term is a reconstruction error as it measures the likelihood $p_{\theta}(\mathbf{x} | \mathbf{z})$ when $\mathbf{z} | \mathbf{x}$. \
\
An important thing to notice is the following: we cannot compute the expectation of the reconstruction loss in a closed form, because we would have to compute an integral.
Integrals estimation is generally non tractable, meaning it requires algorithms which scale exponentially with dimension (generally these are variations of quadrature formulae).
To estimate this integral instead we use a Monte Carlo approach. 
Given the input $\mathbf{x}$, we sample $\mathbf{z}^{l} \ \forall \ l=1,...,\text{L}$ from $q_{\phi}(\mathbf{z} | \mathbf{x})$.
Then, our Monte Carlo estimator is:
```math
\mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})}[ log \text{ } p_{\theta}( \mathbf{x} | \mathbf{z}) ] \approx \frac{1}{L} \sum_{l=1}^{L} log \ p_{\theta}( \mathbf{x} | \mathbf{z}^{l})
```
In the experiments, it is common to use L = 1 and this still yields an estimator with sufficiently low variance.
Furthermore, if we set $q_{\phi} \sim \mathcal{N}(\mathbf{z} | \mathbf{x})$, we can compute the KL divergence in closed form.

# How is the Variational Auto Encoder implemented?
The encoder and decoder networks of the variational auto encoder have to output a conditional distribution over $\mathbf{x}$ and $ \mathbf{z}$.
To achieve this, the networks output the scale and location parameters of the Normal distributions.
For example, if $\mathbf{x}$ is a 3 x 256 x 256 tensor representing an image, the decoder outputs a flattened vector of dimension 3\*256\*256 for the location and another one for the scale (in case you are using an isotropic gaussian, as in the original implementation).

# Why is the reparametrization trick necessary?
The reparametrisation trick is necessary to make sure that the Monte Carlo estimator is differentiable. 
In our case, $\mathcal{N}(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mathbf{z}; \mu(\mathbf{x}), \sigma(\mathbf{x})^2)$.
If one were to sample $\mathbf{z}$ directly from this distribution, it would not be possible to propagate the gradients back to the encoder network, which outputs the values for $\mu$ and $\sigma$.
Instead, if one computes equivalently $\mathbf{z} = \mu(\mathbf{x}) + \sigma(\mathbf{x}) \epsilon$, where $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, this is now differentiable with respect to $\mu$ and $\sigma$ even though there is a sampling operation.

# What are the limitations of the Variational Auto Encoder?
Clearly, one of the main limitations is that you have to explicitly define the functional form of the distributions you use to approximate the real data distribution $p$.
This limits the expressivity of the model. 
I think of the encoder of the VAE in similar terms of Mixture Density Networks. 
While the MDN explicitly define a GMM for every point in the space, the VAE only defines a Gaussian per point in the space. 
However in both cases your budget of Gaussian distributions is infinite as there are infinitely many points on the space. 
Therefore it looks like you should be able to approximate any sort of distribution, if you consider the distribution of the latent variable. 
Of course, the main issue of VAE is that at decoding time, you cannot represent any distribution you want, but rather you are stuck with a Gaussian again. 
Hence there still is some loss of flexibility in the functional form of the VAE.

# How does the VQ-VAE work?
The VQ-VAE introduces a vector quantization operation, with the aim of discretizing the values of the latent variable. This completely changes the probabilities, that rather then being normal distribution, now become a Delta for the encoder, while for the decoder in theory they still consider a gaussian but with fixed unitary covariance. Under these assumptions, the reconstruction loss can be approximated just as an MSE. Since the quantisation operation is not differentiable, the gradient would be lost moving from the decoder to the encoder. Hence, the pass-through operation is used, and it transfers whatever gradient the quantised latent has to the non quantised version. I suspect that also the selected code receives some gradient from this. Anyways, to ensure that the code book is updated, the authors include two additional losses, which are specular: this is a L2 loss between the output of the encoder and the closest code, where the loss is split in two and symmmetrically applies the stop gradient.Apparently this is to simulate the alternating projections algorithm (or for example k-means) where the points and the centroids are updated sequentially, as a joint update might converge worse. Additionally one could control the codebook using a moving average and slow down the convergence by using different weighing parameters for each loss. 

