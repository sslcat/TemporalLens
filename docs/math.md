# What the model computes

## Pixels to frame embeddings

I sample at $t_i=i/10$ seconds using the frame displayed at that time. After rescaling to $W=96,H=64$, I form a mask $M=\{(u,v):R<130,\ G>150,\ B>145\}$. The embedding is

$$
x_i=\left(\frac{\sum_M u}{W|M|},\frac{\sum_M v}{H|M|}\right),\qquad c_i=2(x_i-0.5).
$$

I check the mask area and its occupancy of the tight bounding rectangle. At inference a failed observation anywhere rejects the clip. At training time the generator verifies every observation. The model never receives color, seed or timestamp as an independent predictor. [Encoder](../temporallens/media.py)

## Spatial and temporal heads

For the frame head I use $h_i=c_i$. For the temporal head I clamp neighbor indices to the clip edges and construct

$$
h_i=[c_i,\ \{32(c_{i+j}-c_i)\}_{j=-2}^{2}].
$$

I flatten the five horizontal offsets first, then the five vertical offsets. The resulting 12 channels include two identically zero center-offset channels. A learned affine map of this local window is a reparameterized five-tap temporal convolution. The explicit differences make velocity easier to fit. Both heads apply $\ell_i=W_3\,\mathrm{ReLU}(W_2\,\mathrm{ReLU}(W_1 h_i+b_1)+b_2)+b_3$ with two 32-unit layers and five output classes. The frame head has $32(2+1)+32(32+1)+5(32+1)=1317$ parameters. Using 12 inputs gives 1637 parameters for the temporal head. No attention or recurrent state is used. [Heads](../temporallens/model.py)

## Loss, pooling and localization

For the frame label $y_i$, softmax $p_{ik}=\exp(\ell_{ik})/\sum_j\exp(\ell_{ij})$ and training class frequency $n_k$, I set $w_k=N/(5n_k)$. Within a batch I minimize

$$
L=-\frac{\sum_i w_{y_i}\log p_{i,y_i}}{\sum_i w_{y_i}}.
$$

For recognition given an annotated window $I$, I average its logits and take $\arg\max_{k\in\{1,2,3,4\}} |I|^{-1}\sum_{i\in I}\ell_{ik}$. This oracle-window task assumes a movement occurred. For full-clip localization I take the per-tick argmax over **all five classes**, join identical adjacent labels, discard still and remove runs shorter than three ticks. A run $[a,b)$ becomes $[a/10,b/10)$ seconds, clipped to duration. The displayed score averages $p_{ik}$ over the retained run. [Training](../temporallens/train.py) · [Segment decoder](../temporallens/model.py)

## Measurements

I count oracle-window predictions in a four-by-four confusion matrix $C$ with true classes on rows. Accuracy is $\mathrm{tr}(C)/\sum C$. For each class, $F1_k=2C_{kk}/(\sum_j C_{kj}+\sum_i C_{ik})$. Macro F1 averages the four class scores. Ordered-list accuracy compares the entire predicted direction list in time order to the two annotated directions. The unordered check sorts both lists and retains multiplicity.

For each reference interval $A$ I use the maximum $|A\cap B|/|A\cup B|$ over predicted intervals $B$ of the same class. A missing match contributes zero. For the best positive-overlap match I average the absolute start and end errors in seconds. The report also gives the number of matched events. IoU alone does not fully penalize extra detections, so I report exact-list accuracy alongside it.

I bootstrap whole scene indices with replacement, recomputing the aggregate confusion matrix and the clip averages on each resample. The 2.5th and 97.5th percentiles give the reported intervals. Both heads share each draw for the paired differences. The frame-order interventions preserve the feature multiset. The evaluation code also swaps whole bouts and tests the resulting order end to end. [Evaluation implementation](../temporallens/evaluate.py)

## The IPN Hand experiment

At tick $t$ I letterbox the RGB image to $224\times224$ and normalize channels using the frozen encoder's means and standard deviations. The $16\times16$ patch grid has 384 channels. I average each quadrant and concatenate to obtain $z_t\in\mathbb{R}^{1536}$. I standardize each component with training-only statistics and a .015 floor on its standard deviation. The frame MLP sees $z_t$. The pooled classifier sees $\frac1{15}\sum_{j=-7}^7 z_{t+j}$. Each window mean is invariant to permutations inside that window. Moving a frame between overlapping windows can change dense outputs.

I use this recurrence for temporal layer $\ell$ with dilation $`d_\ell\in\{1,2,4\}`$.

```math
h^{\ell}_t=\mathrm{ReLU}(h^{\ell-1}_t+b_\ell+\sum_{j=-1}^1 W_{\ell j}h^{\ell-1}_{t+jd_\ell})
```

The temporal receptive field is $1+2(1+2+4)=15$ ticks. I replicate input edges and supply context on both sides of training chunks. The five-way weighted cross-entropy uses $w_c=\sqrt{N/n_c}$ from train ticks and divides by the observed target-weight sum. The logistic flow baseline adds $\lambda\lVert W\rVert_F^2/2$ to that loss. [Heads](../temporallens/gesture.py) · [Fitting code](../research/ipn_train.py)

For supplied true interval $I$, the recognition decision maximizes the average log probability over $I$ among four targets. For unknown boundaries I find runs of consecutive above-threshold ticks, include the other class in each argmax, and match predicted/reference intervals one-to-one at each IoU cutoff. Segment F1 is $2TP/(2TP+FP+FN)$. In the AB/BA test the forced assignment compares $s_A(a)+s_B(b)$ to $s_A(b)+s_B(a)$, where $s$ is a bout's mean logit and $a,b$ are the two named targets. A tie receives half credit. Reversing the twin order preserves all input frames. [Metric implementation](../research/ipn_metrics.py) · [Controls](../research/ipn_evaluate.py)

## Two-stage intervals and rejection

Let $u_t$ contain the standardized 5- and 15-tick means of motion features and CNN logits. The binary detector uses $q_t=\sigma(w^\top u_t+b)$. Thresholding $q_t$ at .40, bridging short gaps and filtering short runs produces candidate interval $I$. For target logits $\ell_{tk}$ and target classes $k=1,\ldots,4$, the selected classifier computes

$$
s_{Ik}=\frac{1}{|I|}\sum_{t\in I}\ell_{tk},\qquad
p_{Ik}=\frac{\exp(s_{Ik})}{\sum_{j=1}^{4}\exp(s_{Ij})}.
$$

I accept the winning label when $\max_k p_{Ik}\geq 0.60$. This four-way score assumes a target within the interval. In the validation-only diagnostic, a separate gate estimated candidate presence $g_I$. Its five joint scores were $1-g_I$ for other and $g_I p_{Ik}$ for each target. I excluded this diagnostic rule from the app after it failed the stopping criterion. [Implementation](../temporallens/two_stage.py) · [Calibration measurements](candidate-rejection.md)

For class-blind proposal recall, I divide the number of one-to-one interval matches by the number of annotated targets. A class-aware match also requires the same gesture label. A wrong label can therefore add both a false alarm and a miss. Candidate-level recall in the rejection diagnostic uses only reference gestures already matched by the fixed detector. Full-stream recall includes every reference gesture. [Matching](../research/ipn_metrics.py) · [Operating points](candidate-rejection.md)
