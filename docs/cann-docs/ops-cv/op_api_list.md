# 算子接口（aclnn）

## 使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在`${INSTALL_DIR}/include/aclnnop`，库文件默认在`${INSTALL_DIR}/lib64`，具体文件如下：

- 头文件：①方式1 （推荐）：引用算子仓总头文件aclnn\_ops\_\$\{ops\_project\}.h。②方式2：引用单个算子API的头文件aclnn\_\*.h。
- 库文件：引用算子仓对应的库文件libopapi_${ops_project}.so。注意，原所有算子仓总库文件libopapi.so后续会废弃，不推荐使用，也不支持与单个算子仓库文件同时使用。

${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓名（如math、nn、cv、transformer），请改为实际算子仓名。

## 接口列表

> **确定性简介**：
>
> - 配置说明：因CANN或NPU型号不同等原因，可能无法保证同一个算子多次运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分算子接口可通过`aclrtCtxSetSysParamOpt`（参见[《Runtime运行时 API》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/aolapi/operatorlist_00001.html)）开启确定性算法，使多次运行结果一致。
> - 性能说明：同一个算子采用确定性计算通常比非确定性慢，因此模型单次运行性能可能会下降。但在实验、调试调测等需要保证多次运行结果相同来定位问题的场景，确定性计算可以提升效率。
> - 线程说明：同一线程中只能设置一次确定性状态，多次设置以最后一次有效设置为准。有效设置是指设置确定性状态后，真正执行了一次算子任务下发。如果仅设置，没有算子下发，只能是确定性变量开启但未下发给算子，因此不执行算子。
>   解决方案：暂不推荐一个线程多次设置确定性。该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。

算子接口列表如下：

| 接口名      | 说明     | 确定性说明（A2/A3） | 确定性说明（Atlas 350 加速卡） |
| -------------- | --------------------------- | --------------------------- | --------------------------- |
| [aclnnMrgbaCustom](aclnnMrgbaCustom.md) | 完成张量rgb和张量alpha的透明度乘法计算。 |默认确定性实现| - |
| [aclnnBackgroundReplace](aclnnBackgroundReplace.md) | 将输入的新的背景图片与已有图片进行融合，通过掩码的方式将背景替换为新的背景。 |默认确定性实现|- |
| [aclnnBlendImagesCustom](aclnnBlendImagesCustom.md) | 完成张量rgb、frame和alpha的透明度乘法计算。 |默认确定性实现| - |
| [aclnnGridSampler2D](aclnnGridSampler2D.md) | 根据网格定义的坐标，从输入张量中采样像素值并重映射到输出空间。 |默认确定性实现|默认确定性实现|
| [aclnnGridSampler3D](aclnnGridSampler3D.md) | 根据网格定义的坐标，从输入张量中采样像素值并重映射到输出空间。 |默认确定性实现|默认确定性实现|
| [aclnnGridSampler2DBackward](aclnnGridSampler2DBackward.md) | [aclnnGridSampler2D](aclnnGridSampler2D.md)的反向传播，完成张量input与张量grid的梯度计算。 |默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
| [aclnnGridSampler3DBackward](aclnnGridSampler3DBackward.md) | [aclnnGridSampler3D](aclnnGridSampler3D.md)的反向传播，完成张量input与张量grid的梯度计算。 |默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
| [aclnnRasterizer](aclnnRasterizer.md) | 实现光栅化计算。根据给定的三维空间中的点和面，获取屏幕中每个像素点的最小深度及其对应的面片索引，并计算该面片的重心坐标透视矫正插值。      |默认确定性实现| - |
| [aclnnResize](aclnnResize.md) | 根据scales调整输入张量的大小。                               |默认确定性实现|默认确定性实现|
| [aclnnThreeInterpolateBackward](aclnnThreeInterpolateBackward.md) | 根据grad_x, idx, weight进行三点插值计算梯度得到grad_y。      |默认非确定性实现，不支持配置开启| - |
| [aclnnUpsampleNearest1d](aclnnUpsampleNearest1d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest2d](aclnnUpsampleNearest2d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleTrilinear3d](aclnnUpsampleTrilinear3d.md) | 对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。 |默认确定性实现| - |
| [aclnnUpsampleBicubic2d](aclnnUpsampleBicubic2d.md) | 对由多个输入通道组成的输入信号应用2D双三次上采样。           |默认确定性实现|默认确定性实现|
| [aclnnUpsampleBicubic2dAA](aclnnUpsampleBicubic2dAA.md) | 对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleBicubic2dAAGrad](aclnnUpsampleBicubic2dAAGrad.md) | [aclnnUpsampleBicubic2dAA](aclnnUpsampleBicubic2dAA.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleBicubic2dBackward](aclnnUpsampleBicubic2dBackward.md) | [aclnnUpsampleBicubic2d](aclnnUpsampleBicubic2d.md)的反向传播。 |默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
| [aclnnUpsampleBilinear2d](aclnnUpsampleBilinear2d.md) | 对由多个输入通道组成的输入信号应用2D双线性上采样。|默认确定性实现|默认确定性实现|
| [aclnnUpsampleBilinear2dAA](aclnnUpsampleBilinear2dAA.md) | 对由多个输入通道组成的输入信号应用2D双线性抗锯齿采样。|默认确定性实现|默认确定性实现|
| [aclnnUpsampleBilinear2dAABackward](aclnnUpsampleBilinear2dAABackward.md) | [aclnnUpsampleBilinear2dAA](aclnnUpsampleBilinear2dAA.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleBilinear2dBackward](aclnnUpsampleBilinear2dBackward.md) | [aclnnUpsampleBilinear2d](aclnnUpsampleBilinear2d.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleBilinear2dBackwardV2](aclnnUpsampleBilinear2dBackwardV2.md) | [aclnnUpsampleBilinear2d](aclnnUpsampleBilinear2d.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleLinear1d](aclnnUpsampleLinear1d.md) | 对由多个输入通道组成的输入信号应用线性插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleLinear1dBackward](aclnnUpsampleLinear1dBackward.md) | [aclnnUpsampleLinear1d](aclnnUpsampleLinear1d.md)的反向传播。 |默认确定性实现|默认非确定性实现，支持配置开启|
| [aclnnUpsampleNearestExact1d](aclnnUpsampleNearestExact1d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现| - |
| [aclnnUpsampleNearestExact2d](aclnnUpsampleNearestExact2d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现| - |
| [aclnnUpsampleNearest1dBackward](aclnnUpsampleNearest1dBackward.md) | [aclnnUpsampleNearestExact1d](aclnnUpsampleNearestExact1d.md)的反向传播。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest2dBackward](aclnnUpsampleNearest2dBackward.md) | [aclnnUpsampleNearestExact2d](aclnnUpsampleNearestExact2d.md)的反向传播。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest1dV2](aclnnUpsampleNearest1dV2.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest2dV2](aclnnUpsampleNearest2dV2.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest3d](aclnnUpsampleNearest3d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearest3dBackward](aclnnUpsampleNearest3dBackward.md) | [aclnnUpsampleNearest3d](aclnnUpsampleNearest3d.md)的反向传播。 |默认确定性实现|默认确定性实现|
| [aclnnUpsampleNearestExact1dBackward](aclnnUpsampleNearestExact1dBackward.md) | [aclnnUpsampleNearestExact1d](aclnnUpsampleNearestExact1d.md)的反向传播。 |默认确定性实现| - |
| [aclnnUpsampleNearestExact2dBackward](aclnnUpsampleNearestExact2dBackward.md) | [aclnnUpsampleNearestExact2d](aclnnUpsampleNearestExact2d.md)的反向传播。 |默认确定性实现| - |
| [aclnnUpsampleNearestExact3d](aclnnUpsampleNearestExact3d.md) | 对由多个输入通道组成的输入信号应用最近邻插值算法进行上采样。 |默认确定性实现| - |
| [aclnnUpsampleNearestExact3dBackward](aclnnUpsampleNearestExact3dBackward.md) | [aclnnUpsampleNearestExact3d](aclnnUpsampleNearestExact3d.md)的反向传播。 |默认确定性实现| - |
| [aclnnUpsampleTrilinear3dBackward](aclnnUpsampleTrilinear3dBackward.md) | [aclnnUpsampleTrilinear3d](aclnnUpsampleTrilinear3d.md)的反向传播。 |默认确定性实现| - |
| [aclnnCIoU](aclnnCIoU.md)          | 用于边界框回归的损失函数，在IoU的基础上同时考虑了中心点距离、宽高比和重叠面积，以更全面地衡量预测框与真实框之间的差异。 |默认确定性实现|默认确定性实现|
| [aclnnIou](aclnnIou.md)          | 计算两组矩形框（预测框bBox与真值框gtBox）的交并比（IOU）或前景交叉比（IOF），用于评估其重叠程度。 |默认确定性实现|默认确定性实现|
| [aclnnNonMaxSuppression](aclnnNonMaxSuppression.md)          | 删除分数小于scoreThreshold的边界框，筛选出与之前被选中部分重叠较高（IOU较高）的框。 |默认确定性实现|- |
| [aclnnRoiAlign](aclnnRoiAlign.md) | RoIAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。 |默认确定性实现|- |
| [aclnnRoiAlignV2](aclnnRoiAlignV2.md) | RoIAlign是一种池化层，用于非均匀输入尺寸的特征图，并输出固定尺寸的特征图。 |默认确定性实现|- |
| [aclnnRoiAlignV2Backward](aclnnRoiAlignV2Backward.md) |[aclnnRoiAlignV2](aclnnRoiAlignV2.md)的反向传播。 |默认非确定性实现，支持配置开启|- |
| [aclnnRoiPoolingWithArgMax](aclnnRoiPoolingWithArgMax.md) | 对输入特征图按 ROI（感兴趣区域）进行池化，在每个 ROI 内按空间划分为 pooled_h × pooled_w 个格子，对每个格子做最大池化，并输出池化结果及最大值在通道内的一维索引（argmax）。|默认确定性实现|默认确定性实现|
| [aclnnRoiPoolingGradWithArgMax](aclnnRoiPoolingGradWithArgMax.md) | [aclnnRoiPoolingWithArgMax](aclnnRoiPoolingWithArgMax.md)的反向传播。 |默认确定性实现|默认非确定性实现，不支持配置开启|
| [aclnnIm2colBackward](aclnnIm2colBackward.md) | 从批处理输入张量中提取滑动局部块，将滑动局部块数组合并为一个大张量。 |默认确定性实现|默认确定性实现|
