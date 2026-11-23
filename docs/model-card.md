# Model card

I trained TemporalLens v1 to label still, left, right, up and down for one cyan marker. The released predictor is a 1,637-parameter five-frame temporal CNN with a fixed two-coordinate pixel encoder. The companion 1,317-parameter frame checkpoint provides the experimental comparison. Both are PyTorch state dictionaries. The release loads the architecture from source and reads tensor weights with `weights_only=True`.

I trained from random initialization with weighted frame-level cross-entropy on 240 generated clips and selected checkpoints on 60 validation clips. The lowest validation loss occurred at epoch 13 for the frame head and 140 for the temporal head. I evaluated the selected heads once in the preregistered comparison on 96 additional scenes. [Training](training.md) · [Dataset](dataset-card.md) · [Results and interventions](evaluation.md)

The supported input is a continuous view of one visible, bright cyan marker undergoing cardinal translations with pauses against a dark background. The model can identify runs outside the two-event training pattern, but those patterns have no benchmark coverage. A geometric marker-quality check rejects some unsupported inputs. It cannot establish semantic compatibility for every uploaded scene. A similarly colored patch, an edit, a diagonal trajectory or a camera pan can yield confident incorrect events. The softmax display score is uncalibrated. I have no evidence for human action recognition, open-world event detection, planning or causal reasoning.

I chose MIT for the original checkpoints and generated assets. No pretrained weights or external video dataset enter the model. I used AI coding assistance for implementation, experiment execution and documentation. [Attribution](../NOTICE.md)

## Experimental IPN Hand checkpoint

I use frozen DINOv2-S/14 features and a 135,749-parameter temporal CNN for four prompted hand gestures. The public head is the prespecified seed-17 checkpoint at epoch 13. It trained on 32 recordings, was selected on eight validation recordings and was evaluated on 16 recordings from eight held-out participants. It uses offline context on both sides of each prediction. Scores are uncalibrated. Three consecutive 7.5 Hz ticks must each reach 0.65 for the same winning target. A perfectly static decoded upload also returns an empty list.

I would use this mode to inspect model behavior, with a person checking the underlying footage. On full held-out streams the detector made 93 unmatched predictions and matched two of 64 true events at IoU ≥.50. Other hand motions, unfamiliar cameras and low visibility can produce confident mistakes. It performs no validated open-set recognition. A blank result means that nothing passed the decoder. [Pilot evaluation](ipn-results.md) · [Weights and attribution](../artifacts/ipn/NOTICE.md)
