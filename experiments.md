# Experiments

Version 2.0

* New data loader
* New evaluator

## UNet, LiTS

| config                        | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | eval        | remark |
| ----------------------------- | ------ | ------ | ------ | ------ | ------ | ----------- | ------ |
| 001_unet_noise_0_05           | 0.953  | 0.964  | 0.968  | 0.932  | 0.959  | mirror+best | Liver  |
|                               | 0.527  | 0.613  | 0.691  | 0.597  | 0.522  | mirror+best | Tumor  |
| 001_unet_noise_0_05_f1_lrp_40 |        | 0.960  |        |        |        |             |        |
|                               |        | 0.590  |        |        |        |             |        |
|                               |        |        |        |        |        |             |        |
|                               |        |        |        |        |        |             |        |



## Augmentation: Noise

| fold 2 | 0.03          | 0.05  | 0.07          |
| ------ | ------------- | ----- | ------------- |
| Liver  | 0.968         | 0.968 | 0.963         |
| Tumor  | 0.680 (final) | 0.691 | 0.664 (final) |

## Augmentation: Flip

| fold 1 | Test | no flip | flip left/right | flip up/down | flip left/right up/down |
| ------ | ---- | ------- | --------------- | ------------ | ----------------------- |
| Liver  | Off  |         |                 |              |                         |
|        | on   | ---     | 0.964           |              | 0.957                   |
| Tumor  | Off  |         |                 |              |                         |
|        | on   | ---     | 0.613           |              | 0.619                   |

