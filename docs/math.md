# What the model computes

## Pixels to frame embeddings

I sample at $t_i=i/10$ seconds using the frame displayed at that time. After rescaling to $W=96,H=64$, I form a mask $M=\{(u,v):R<130,\ G>150,\ B>145\}$. The embedding is

$$x_i=\left(\frac{\sum_M u}{W|M|},\frac{\sum_M v}{H|M|}\right),\qquad c_i=2(x_i-0.5).$$

I check the mask area and its occupancy of the tight bounding rectangle. At inference a failed observation anywhere rejects the clip. At training time the generator verifies every observation. The model never receives color, seed or timestamp as an independent predictor. [Encoder](../temporallens/media.py)

## Spatial and temporal heads

For the frame head I use $h_i=c_i$. For the temporal head I clamp neighbor indices to the clip edges and construct

$$h_i=[c_i,\ \{32(c_{i+j}-c_i)\}_{j=-2}^{2}].$$

The resulting 12 channels include two identically zero center-offset channels. A learned affine map of this local window is a reparameterized five-tap temporal convolution. The explicit differences make velocity easier to fit. Both heads apply $\ell_i=W_3\,\mathrm{ReLU}(W_2\,\mathrm{ReLU}(W_1 h_i+b_1)+b_2)+b_3$ with two 32-unit layers and five output classes. The frame head has $32(2+1)+32(32+1)+5(32+1)=1317$ parameters. Using 12 inputs gives 1637 parameters for the temporal head. No attention or recurrent state is used. [Heads](../temporallens/model.py)

## Loss, pooling and localization

For the frame label $y_i$, softmax $p_{ik}=\exp(\ell_{ik})/\sum_j\exp(\ell_{ij})$ and training class frequency $n_k$, I set $w_k=N/(5n_k)$. Within a batch I minimize

$$L=-\frac{\sum_i w_{y_i}\log p_{i,y_i}}{\sum_i w_{y_i}}.$$

For recognition given an annotated window $I$, I average its logits and take $\arg\max_{k\in\{1,2,3,4\}} |I|^{-1}\sum_{i\in I}\ell_{ik}$. This oracle-window task assumes a movement occurred. For full-clip localization I take the per-tick argmax over **all five classes**, join identical adjacent labels, discard still and remove runs shorter than three ticks. A run $[a,b)$ becomes $[a/10,b/10)$ seconds, clipped to duration. The displayed score averages $p_{ik}$ over the retained run. [Training](../temporallens/train.py) · [Segment decoder](../temporallens/model.py)

## Measurements

I count oracle-window predictions in a four-by-four confusion matrix $C$ with true classes on rows. Accuracy is $\mathrm{tr}(C)/\sum C$. For each class, $F1_k=2C_{kk}/(\sum_j C_{kj}+\sum_i C_{ik})$. Macro F1 averages the four class scores. Ordered-list accuracy compares the entire predicted direction list in time order to the two annotated directions. The unordered check sorts both lists and retains multiplicity.

For each reference interval $A$ I use the maximum $|A\cap B|/|A\cup B|$ over predicted intervals $B$ of the same class. A missing match contributes zero. For the best positive-overlap match I average the absolute start and end errors in seconds. The report also gives the number of matched events. IoU alone does not fully penalize extra detections, so I report exact-list accuracy alongside it.

I bootstrap whole scene indices with replacement, recomputing the aggregate confusion matrix and the clip averages on each resample. The 2.5th and 97.5th percentiles give the reported intervals. Both heads share each draw for the paired differences. The frame-order interventions preserve the feature multiset. The evaluation code also swaps whole bouts and tests the resulting order end to end. [Evaluation implementation](../temporallens/evaluate.py)
