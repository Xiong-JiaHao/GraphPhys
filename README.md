## GraphPhys: Facial Video-based Physiological Measurement with Graph Neural Network
* Color Space：RGB
* Dataset：VIPL
* Model_TDTransformer
  * physiological signal extraction module：TDTransformer(PhysFormer)
  * graph denoising module：KNN + ARConv + Conv1d(768, 192, 96, 24, 1)
  * physiological indicator estimation module：MLP(160->280->140)
* Model_PhysNet
  * physiological signal extraction module：PhysNet(PhysNet)
  * graph denoising module：MLP(4096->768) + KNN + ARConv + Conv1d(768, 192, 96, 24, 1)
  * physiological indicator estimation module：MLP(160->280->140)
* Loss： NegPearson + MultiFocalLoss(hr_class[40->180])
