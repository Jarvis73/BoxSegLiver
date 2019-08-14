

# Experiments

Version 2.0

* New data loader
* New evaluator

## UNet, LiTS

| config                          | fold 0 | fold 1 | fold 2      | fold 3 | fold 4 | eval        | remark          |
| ------------------------------- | ------ | ------ | ----------- | ------ | ------ | ----------- | --------------- |
| 001_unet_noise_0_05             | 0.953  | 0.964  | 0.968       | 0.932  | 0.959  | mirror+best | Liver           |
|                                 | 0.527  | 0.613  | **0.691**   | 0.597  | 0.522  | mirror+best | Tumor           |
| 001_unet_noise_0_05_f1_lrp_40   |        | 0.960  |             |        |        |             |                 |
|                                 |        | 0.590  |             |        |        |             |                 |
| 001_unet_noise_0_05_f2_decay_03 |        |        | 0.966       |        |        |             |                 |
|                                 |        |        | 0.685       |        |        |             |                 |
| 001_unet_rflip                  |        | 0.957  | 0.969       |        |        |             |                 |
|                                 |        | 0.619  | 0.661       |        |        |             |                 |
| 011_gnet_de_lr                  |        |        | 0.970       |        |        |             |                 |
|                                 |        |        | 0.716       |        |        |             |                 |
| 011_gnet_de_lrv2 (plateau)      |        |        | 0.970       |        |        |             |                 |
|                                 |        |        | 0.736       |        |        |             |                 |
| 011_gnet_de_rflip               | 0.959  | 0.968  | 0.970       | 0.939  |        |             |                 |
|                                 | 0.642  | 0.664  | **0.748**   | 0.675  |        |             |                 |
| 012_gnet_sp_lr                  |        |        | 0.970       |        |        |             |                 |
|                                 |        |        | 0.771       |        |        |             |                 |
| 012_gnet_sp_lr_v2 (global dice) |        |        | 0.970       |        |        |             |                 |
|                                 |        |        | **0.783**   |        |        |             |                 |
| 012_gnet_sp_rflip               |        |        | 0.970       |        |        |             |                 |
|                                 |        |        | 0.764       |        |        |             |                 |
| 012_gnet_sp_lr_v3               |        |        | 0.970       |        |        |             |                 |
|                                 |        |        | 0.777       |        |        |             |                 |
| 013_gnet_sp_rand                |        |        | 0.970       |        |        |             |                 |
|                                 |        |        | 0.723/0.749 |        |        |             | no/middle guide |



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

## 012_gnet_sp_lr_v2 --> User global dice as metric

| checkpoint_best | Liver | Tumor | Liver-Mirror-1 | Tumor-Mirror-1 |
| --------------- | ----- | ----- | -------------- | -------------- |
| 100000          | 0.966 | 0.749 |                |                |
| 200000          | 0.970 | 0.751 |                |                |
| 300000          | 0.969 | 0.766 |                |                |
| 400000          | 0.970 | 0.772 | 0.970          | 0.773          |
| 500000          | 0.970 | 0.754 | 0.970          | 0.776          |
| 600000          | 0.970 | 0.758 | 0.970          | **0.783**      |
| 700000          |       |       | 0.970          | 0.777          |

## Notes

{'contrast': 1.2692871093749996, 'dissimilarity': 1.0822916666666664, 'homogeneity': 2.0, 'energy': 2.0, 'entropy': 1.1006214881102814, 'correlation': 1.0, 'cluster_shade': 6.55612959564772, 'cluster_prominence': 43.42668102560059}                                                                                                                                                                                               {'contrast': 0.0, 'dissimilarity': 0.0, 'homogeneity': 0.0, 'energy': 0.0, 'entropy': -0.0, 'correlation': -0.9618304945618715, 'cluster_shade': -9.778071268534974, 'cluster_prominence': 0.0}

#### Test

{'contrast': 0.5841552734375, 'dissimilarity': 0.6072916666666665, 'homogeneity': 1.0922337042925268, 'energy': 0.9079717167408267, 'entropy': 1.085120912736222, 'correlation': 0.9924501352098303, 'cluster_shade': 3.647052967014026, 'cluster_prominence': 30.635456156441723}                                                                                                                                                    {'contrast': 0.0007273356119791664, 'dissimilarity': 0.01985677083333333, 'homogeneity': 0.006696220595940001, 'energy': 0.029236633069144737, 'entropy': 0.3106133312234999, 'correlation': -0.8858143607705778, 'cluster_shade': -6.1290989137606084, 'cluster_prominence': 1.1767866089940076e-05}