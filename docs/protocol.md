# Motion experiment • protocol v1

I fixed these decisions before fitting either model. My unit of replication is an independently rendered scene. I generate 240 training, 60 validation and 96 test clips. Each split balances the 12 ordered pairs of distinct cardinal translations. Each eight-second clip contains two bouts of movement with stationary intervals. All frames, appearances and annotations come from the same scene seed. No people or source videos are reused.

**H1.** A five-frame temporal convolution will exceed the independent-frame head by at least 0.15 in four-way macro F1 on annotated event windows. Both heads receive identical per-frame marker positions, training clips, loss weights, optimization budget and checkpoint selection. Event recognition averages direction logits over a true interval. It receives privileged boundaries.

**H2.** The temporal head will gain at least 0.20 in exact ordered-list accuracy using predicted boundaries. Extras, omissions and direction errors each invalidate the list. I also report unordered multiset accuracy to distinguish order errors from detection errors.

**H3.** On reversed videos the temporal head will recognize at least 75% of direction-inverted events. At fixed original targets, alignment back to original indices after reversal or random permutation should depress temporal recognition by at least 0.20 relative to clean accuracy. The frame head should be invariant after undoing either permutation. Reversal uses a half-open mirrored time window, an approximation within one frame of transition semantics.

I fix the segment decoder to discard runs shorter than 3 samples. For localization I average the best same-class temporal IoU per true event, assigning zero to misses. I measure endpoint MAE only for same-class overlaps. Each true event has a distinct label in its clip. Confusion matrices and class F1 use oracle windows. I report paired clip bootstrap 95% intervals for primary metrics and differences using 2,000 replicates. These quantify held-out scene sampling, conditional on one training seed.

I select the lowest validation weighted cross-entropy checkpoint within 140 epochs per model. I record synchronized batch-one model latency, decode-plus-inference latency, peak allocated CUDA training memory and elapsed fit time. I will inspect best and worst clips by localization and document any failure to support these hypotheses. No test-based retuning is part of this run.
