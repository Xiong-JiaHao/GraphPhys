## GraphPhys: Facial Video-base Physiological Measurement with Graph Neural Network
* 颜色空间：RGB
* 数据集：VIPL
* Model_TDTransformer
  * 生理信号生成网络：TDTransformer(PhysFormer)
  * 波形聚合图神经网络：KNN + ARConv + Conv1d(768, 192, 96, 24, 1)
  * 心率估计网络：MLP(160->280->140)
* Model_PhysNet
  * 生理信号生成网络：PhysNet(PhysNet)
  * 波形聚合图神经网络：MLP(4096->768) + KNN + ARConv + Conv1d(768, 192, 96, 24, 1)
  * 心率估计网络：MLP(160->280->140)
* 损失函数： NegPearson + MultiFocalLoss(hr_class[40->180])
