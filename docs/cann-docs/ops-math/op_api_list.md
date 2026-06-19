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
> - 符号说明：表中 " - " 符号表示该接口暂不支持当前列产品。

算子接口列表如下：

|    接口名   |   说明     | 确定性说明（A2/A3） | 确定性说明（Atlas 350 加速卡） |
| ----------- | ------------------- | --------- | --------- |
| [aclnnAbs](aclnnAbs.md)     | 实现张量的绝对值运算。    | 默认确定性实现| 默认确定性实现|
| [aclnnAcos&aclnnInplaceAcos](aclnnAcos&aclnnInplaceAcos.md) | 对输入矩阵的每个元素进行反余弦操作后输出。                   | 默认确定性实现| - |
| [aclnnAcosh&aclnnInplaceAcosh](aclnnAcosh&aclnnInplaceAcosh.md) | 对输入Tensor中的每个元素进行反双曲余弦操作后输出。           | 默认确定性实现| - |
| [aclnnAngleV2](aclnnAngleV2.md) | 为输入张量的每一个元素取角度（单位：弧度）。 | 默认确定性实现 | - |
| [aclnnAdd&aclnnInplaceAdd](aclnnAdd&aclnnInplaceAdd.md) | 完成加法计算。                                               | 默认确定性实现| 默认确定性实现|
| [aclnnAddcdiv&aclnnInplaceAddcdiv](aclnnAddcdiv&aclnnInplaceAddcdiv.md) | 完成加法计算。                                               | 默认确定性实现| 默认确定性实现|
| [aclnnAddcmul&aclnnInplaceAddcmul](aclnnAddcmul&aclnnInplaceAddcmul.md) | 执行 tensor1 与 tensor2 的逐元素乘法，将结果乘以标量值value并与输入self/selfRef做逐元素加法。 | 默认确定性实现| 默认确定性实现|
| [aclnnAddLora](aclnnAddLora.md)        | 为神经网络添加LoRA（Low-Rank Adaptation）层功能，通过低秩分解减少参数数量。 | 默认确定性实现| - |
| [aclnnAddr&aclnnInplaceAddr](aclnnAddr&aclnnInplaceAddr.md) | 求一维向量vec1和vec2的外积得到一个二维矩阵，并将外积结果矩阵乘一个系数后和自身乘系数相加后输出。 | 默认确定性实现| 默认确定性实现|
| [aclnnAdds](aclnnAdds.md) | 完成加法计算。 | 默认确定性实现|  默认确定性实现 |
| [aclnnAddV3&aclnnInplaceAddV3](aclnnAddV3&aclnnInplaceAddV3.md) | 完成加法计算。                                               | 默认确定性实现|  默认确定性实现  |
| [aclnnAll](aclnnAll.md)              | 对于给定维度dim中的每一维，如果输入Tensor中该维度对应的所有元素计算为True，则返回True，否则返回False。 | 默认确定性实现| 默认确定性实现|
| [aclnnAmax](aclnnAmax.md)            | 返回张量在指定维度(dim)上每个切片的最大值。                  | 默认确定性实现| 默认确定性实现|
| [aclnnAmin](aclnnAmin.md)            | 返回张量在指定维度(dim)上每个切片的最小值。                  | 默认确定性实现| 默认确定性实现|
| [aclnnAminmax](aclnnAminmax.md)      | 返回输入张量在指定维度上每行的最小值和最大值。               | 默认确定性实现| 默认确定性实现|
| [aclnnAminmaxAll](aclnnAminmaxAll.md) | 返回输入张量在所有维度上的最小值和最大值。                   | 默认确定性实现| 默认确定性实现|
| [aclnnAminmaxDim](aclnnAminmaxDim.md) | 返回输入张量在指定维度上的最小值和最大值。                   | 默认确定性实现| 默认确定性实现|
| [aclnnAny](aclnnAny.md)              | 对于给定维度dim中的每一维，如果输入Tensor中该维度对应的任意元素计算为True，则返回True，否则返回False。 | 默认确定性实现| 默认确定性实现|
| [aclnnArange](aclnnArange.md)             | 从start起始到end结束按照step的间隔获取值，并保存到输出的1维张量，其中数据范围为[start,end)。 | 默认确定性实现|  默认确定性实现  |
| [aclnnArgMax](aclnnArgMax.md)        | 返回张量在指定维度（dim）上的最大值的索引，并保存到out张量中。 | 默认确定性实现| 默认确定性实现|
| [aclnnArgMin](aclnnArgMin.md)           | 返回tensor中指定轴的最小值索引，并保存到out中。              | 默认确定性实现| 默认确定性实现|
| [aclnnAsin&aclnnInplaceAsin](aclnnAsin&aclnnInplaceAsin.md) | 对输入矩阵的每个元素进行反正弦操作后输出。                   | 默认确定性实现| - |
| [aclnnAsinh&aclnnInplaceAsinh](aclnnAsinh&aclnnInplaceAsinh.md) | 对输入Tensor中的每个元素进行反双曲正弦操作后输出。           | 默认确定性实现| - |
| [aclnnAtan&aclnnInplaceAtan](aclnnAtan&aclnnInplaceAtan.md) | 对输入矩阵的每个元素进行反正切操作后输出。                   | 默认确定性实现| 默认确定性实现|
| [aclnnAtan2&aclnnInplaceAtan2](aclnnAtan2&aclnnInplaceAtan2.md) | 对输入张量self和other进行逐元素的反正切运算。                | 默认确定性实现|   |
| [aclnnAtanh&aclnnInplaceAtanh](aclnnAtanh&aclnnInplaceAtanh.md) | 对输入Tensor中的每个元素进行反双曲正切操作后输出。           | 默认确定性实现| - |
| [aclnnBatchNormStats](aclnnBatchNormStats.md) | 计算单卡输入数据的均值和标准差的倒数。                       | 默认确定性实现| 默认确定性实现|
| [aclnnBernoulli&aclnnInplaceBernoulli](aclnnBernoulli&aclnnInplaceBernoulli.md) | 从伯努利分布中提取二进制随机数（0 或 1），prob为生成二进制随机数的概率，输入的张量用于指定shape。  | 默认确定性实现|  默认确定性实现 |
| [aclnnBernoulliTensor&aclnnInplaceBernoulliTensor](aclnnBernoulliTensor&aclnnInplaceBernoulliTensor.md) | 从伯努利分布中提取二进制随机数（0 或 1），prob为生成二进制随机数的概率，输入的张量用于指定shape。   | 默认确定性实现| 默认确定性实现 |
| [aclnnBincount](aclnnBincount.md) | 计算非负整数数组中每个数的频率。                             | 默认确定性实现| 默认非确定性实现，支持配置开启|
| [aclnnBitwiseNot](aclnnBitwiseNot.md) | 输入为BOOL型tensor时，进行逻辑非运算；输入为INT型时进行按位非运算。 | 默认确定性实现| 默认确定性实现|
| [aclnnBitwiseAndScalar](aclnnBitwiseAndScalar.md) | 计算输入tensor中每个元素和输入标量的按位与结果。             | 默认确定性实现| 默认确定性实现|
| [aclnnBitwiseAndTensor](aclnnBitwiseAndTensor.md) | 输入为BOOL型tensor时，进行逻辑与运算；输入为INT型时，进行位与运算。 | 默认确定性实现| 默认确定性实现|
| [aclnnBitwiseAndTensorOut&aclnnInplaceBitwiseAndTensorOut](aclnnBitwiseAndTensorOut&aclnnInplaceBitwiseAndTensorOut.md) | 输入为BOOL型tensor时，进行逻辑与运算；输入为INT型时，进行位与运算。 | 默认确定性实现| 默认确定性实现|
| [aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar](aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar.md) | 计算输入张量self中每个元素和输入标量other的按位或。          | 默认确定性实现| 默认确定性实现|
| [aclnnBitwiseOrTensor&aclnnInplaceBitwiseOrTensor](aclnnBitwiseOrTensor&aclnnInplaceBitwiseOrTensor.md) | 计算张量self中每个元素与other张量中对应位置的元素的按位或。  | 默认确定性实现| 默认确定性实现|
| [aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar](aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar.md) | 计算输入张量self中每个元素和输入标量other的按位异或，输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑异或。 | 默认确定性实现| 默认确定性实现|
| [aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor](aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor.md) | 计算输入张量self中每个元素与输入张量other中对应位置元素的按位异或，输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑异或。 | 默认确定性实现| 默认确定性实现|
| [aclnnCast](aclnnCast.md)                  | 实现张量数据类型转换。                                       | 默认确定性实现| 默认确定性实现|
| [aclnnCat](aclnnCat.md)            | 将tensors中所有tensor按照维度dim进行级联。    | 默认确定性实现| 默认确定性实现  |
| [aclnnCdistBackward](aclnnCdistBackward.md)            | 完成aclnnCdist的反向    | 默认确定性实现| - |
| [aclnnCeil&aclnnInplaceCeil](aclnnCeil&aclnnInplaceCeil.md) | 返回输入tensor中每个元素向上取整的结果。                     | 默认确定性实现| 默认确定性实现|
| [aclnnChannelShuffle](aclnnChannelShuffle.md) | 将(*, C, H, W)张量的channels分成g个组，然后将每个通道组中的通道进行随机重排，最后将所有通道合并输出，同时保持最终输出张量的shape和输入张量保持一致。 | 默认确定性实现| 默认确定性实现|
| [aclnnChunkCat](aclnnChunkCat.md) | 将tensors中所有tensor先按照维度dim切分为numChunks块，再按照dim后一维进行级联，最后转换为out的数据类型。 | 默认确定性实现| - |
| [aclnnCircularPad2d](aclnnCircularPad2d.md) | 使用输入循环填充输入tensor的最后两维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnCircularPad3d](aclnnCircularPad3d.md) | 使用输入循环填充输入tensor的最后三维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnCircularPad2dBackward](aclnnCircularPad2dBackward.md) | circular_pad2d的反向传播, 前向计算参考aclnnCircularPad2d。   | 默认确定性实现| 默认确定性实现|
| [aclnnCircularPad3dBackward](aclnnCircularPad3dBackward.md) | aclnnCircularPad3d的反向传播。                               | 默认确定性实现| 默认确定性实现|
| [aclnnClamp](aclnnClamp.md) | 将输入的所有元素限制在[min,max]范围内，如果min为None，则没有下限，如果max为None，则没有上限。 | 默认确定性实现| 默认确定性实现|
| [aclnnClampMax&aclnnInplaceClampMax](aclnnClampMax&aclnnInplaceClampMax.md) | 将输入的所有元素限制在[-inf,max]范围内。 | 默认确定性实现| 默认确定性实现|
| [aclnnClampMaxTensor&aclnnInplaceClampMaxTensor](aclnnClampMaxTensor&aclnnInplaceClampMaxTensor.md) | 将输入的所有元素限制在[-inf, max]范围内。 | 默认确定性实现| 默认确定性实现|
| [aclnnClampMin](aclnnClampMin.md) | 将输入的所有元素限制在[min, inf]范围内。 | 默认确定性实现| 默认确定性实现|
| [aclnnClampMinTensor&aclnnInplaceClampMinTensor](aclnnClampMinTensor&aclnnInplaceClampMinTensor.md) | 将输入的所有元素限制在[min, inf]范围内。 | 默认确定性实现| 默认确定性实现|
| [aclnnClampTensor](aclnnClampTensor.md) | 将输入的所有元素限制在[min, max]范围内，如果min缺省，则无下限，如果max缺省，则无上限。 | 默认确定性实现| 默认确定性实现|
| [aclnnComplex](aclnnComplex.md)         | 输入两个Shape和Dtype一致的Tensor：real和imag。               | 默认确定性实现| - |
| [aclnnConfusionTranspose](aclnnConfusionTranspose.md) | 融合reshape和transpose运算。 | 默认确定性实现|  默认确定性实现 |
| [aclnnConstantPadNd](aclnnConstantPadNd.md) | 对输入的张量self，以pad参数为基准进行数据填充，填充值为value。 | 默认确定性实现| 默认确定性实现|
| [aclnnCos&aclnnInplaceCos](aclnnCos&aclnnInplaceCos.md) | 对输入矩阵的每个元素进行余弦操作后输出。                     | 默认确定性实现| 默认确定性实现|
| [aclnnCosh&aclnnInplaceCosh](aclnnCosh&aclnnInplaceCosh.md) | 双曲函数，根据公式返回一个新的tensor。结果的形状与输入tensor相同。 | 默认确定性实现| 默认确定性实现|
| [aclnnCummax](aclnnCummax.md)            | 计算self中的累积最大值，并返回最大值以及对应的索引。         | 默认确定性实现| - |
| [aclnnCummin](aclnnCummin.md)            | 计算self中的累积最小值，并返回最小值以及对应的索引。         | 默认确定性实现|   |
| [aclnnCumprod&aclnnInplaceCumprod](aclnnCumprod&aclnnInplaceCumprod.md) | 新增aclnnCumprod接口，cumprod函数用于计算输入张量在指定维度上的累积乘积。 | 默认确定性实现| - |
| [aclnnCumsum](aclnnCumsum.md)            | 对输入张量self的元素，按照指定维度dim依次进行累加，并将结果保存到输出张量out中。 | 默认确定性实现| 默认确定性实现|
| [aclnnCumsumV2](aclnnCumsumV2.md)        | 对输入张量self的元素，按照指定维度dim依次进行累加，并将结果保存到输出张量out中。 | 默认确定性实现| 默认确定性实现|
| [aclnnDiagFlat](aclnnDiagFlat.md) | 生成对角线张量。如果输入self为一维张量，则返回二维张量，self里元素为对角线值；如果输入self是二维及以上张量，则先进行扁平化（化简为一维张量），再转化为第一种场景处理。 | 默认确定性实现| 默认确定性实现|
| [aclnnDiv&aclnnInplaceDiv](aclnnDiv&aclnnInplaceDiv.md) | 完成除法计算。                                               | 默认确定性实现| 默认确定性实现|
| [aclnnDivMod&aclnnInplaceDivMod](aclnnDivMod&aclnnInplaceDivMod.md) | 完成除法计算，并根据mode参数选择舍入操作。    | 默认确定性实现| 默认确定性实现 |
| [aclnnDivMods&aclnnInplaceDivMods](aclnnDivMods&aclnnInplaceDivMods.md) | 完成除法计算，并根据mode参数选择舍入操作。    | 默认确定性实现| 默认确定性实现 |
| [aclnnDivs&aclnnInplaceDivs](aclnnDivs&aclnnInplaceDivs.md) | 完成除法计算。                                               | 默认确定性实现| 默认确定性实现 |
| [aclnnDot](aclnnDot.md)                     | 计算两个一维张量的点积结果。                                 | 默认确定性实现| 默认确定性实现|
| [aclnnDropoutDoMask](aclnnDropoutDoMask.md)         | 按照概率prob随机将输入中的元素置零，并将输出按照1/(1-prob)的比例放大。 | 默认确定性实现| 默认确定性实现  |
| [aclnnDropoutV3](aclnnDropoutV3.md)         | 按照概率p随机将输入中的元素置零，并将输出按照1/(1-p)的比例缩放。 | 默认确定性实现| 默认确定性实现  |
| [aclnnEqScalar&aclnnInplaceEqScalar](aclnnEqScalar&aclnnInplaceEqScalar.md) | 计算self中的元素的值与other的值是否相等，将self每个元素与other的值的比较结果写入out中。 | 默认确定性实现| 默认确定性实现|
| [aclnnEqTensor&aclnnInplaceEqTensor](aclnnEqTensor&aclnnInplaceEqTensor.md) | 计算两个Tensor中的元素是否相等，返回一个Tensor，self=other的为True(1.)，否则为False(0.)。 | 默认确定性实现| 默认确定性实现|
| [aclnnEqual](aclnnEqual.md)               | 计算两个Tensor是否有相同的大小和元素，返回一个Bool类型。     | 默认确定性实现| 默认确定性实现  |
| [aclnnErf&aclnnInplaceErf](aclnnErf&aclnnInplaceErf.md) | 返回输入Tensor中每个元素对应的误差函数的值。                 | 默认确定性实现| 默认确定性实现|
| [aclnnErfc&aclnnInplaceErfc](aclnnErfc&aclnnInplaceErfc.md) | 返回输入Tensor中每个元素对应的误差互补函数的值。             | 默认确定性实现| - |
| [aclnnExp&aclnnInplaceExp](aclnnExp&aclnnInplaceExp.md) | 返回一个新的张量，该张量的每个元素都是输入张量对应元素的指数。 | 默认确定性实现| 默认确定性实现|
| [aclnnExpand](aclnnExpand.md)            | 将输入张量self广播成指定size的张量。                         | 默认确定性实现| 默认确定性实现|
| [aclnnExpm1&aclnnInplaceExpm1](aclnnExpm1&aclnnInplaceExpm1.md) | 以输入的self为指数，计算自然常数e的幂，并对指数计算结果进行减1计算。 | 默认确定性实现| 默认确定性实现|
| [aclnnExp2&aclnnInplaceExp2](aclnnExp2&aclnnInplaceExp2.md) | self每个元素作为基数2的幂完成计算。                          | 默认确定性实现| 默认确定性实现|
| [aclnnExpSegsum](aclnnExpSegsum.md)      | 进行分段和计算。生成对角线为0的半可分矩阵，且上三角为-inf。  | 默认确定性实现| - |
| [aclnnExpSegsumBackward](aclnnExpSegsumBackward.md)      | [aclnnExpSegsum](aclnnExpSegsum.md)的反向计算。  | 默认确定性实现| - |
| [aclnnEye](aclnnEye.md)                     | 返回一个二维张量，该张量的对角线上元素值为1，其余元素值为0。 | 默认确定性实现| 默认确定性实现  |
| [aclnnFlatten](aclnnFlatten.md)   | 将输入Tensor，基于给定的axis，扁平化为一个2D的Tensor。       | 默认确定性实现| 默认确定性实现|
| [aclnnFloor&aclnnInplaceFloor](aclnnFloor&aclnnInplaceFloor.md) | 返回输入Tensor中每个元素向下取整，并将结果回填到输入Tensor中。 | 默认确定性实现| 默认确定性实现|
| [aclnnFloorDivide&aclnnInplaceFloorDivide](aclnnFloorDivide&aclnnInplaceFloorDivide.md) |  完成除法计算，对余数向下取整。                                                            | 默认确定性实现| 默认确定性实现|
| [aclnnFloorDivides&aclnnInplaceFloorDivides](aclnnFloorDivides&aclnnInplaceFloorDivides.md) | 完成除法计算，对余数向下取整。                               | 默认确定性实现| 默认确定性实现|
| [aclnnFmodScalar&aclnnInplaceFmodScalar](aclnnFmodScalar&aclnnInplaceFmodScalar.md) | 返回 self除以other的余数。                                   | 默认确定性实现| 默认确定性实现|
| [aclnnFmodTensor&aclnnInplaceFmodTensor](aclnnFmodTensor&aclnnInplaceFmodTensor.md) | 返回self除以other的余数。                                    | 默认确定性实现| 默认确定性实现|
| [aclnnFrac&aclnnInplaceFrac](aclnnFrac&aclnnInplaceFrac.md) | 计算输入Tensor中每个元素的小数部分后输出。                   | 默认确定性实现|   |
| [aclnnGcd](aclnnGcd.md)                     | 对给定的self和other计算element-wise维度的最大公约数，其中self和other都需要为整数。 | 默认确定性实现| 默认确定性实现|
| [aclnnGer](aclnnGer.md)                     | 实现self和vec2的外积。                                       | 默认确定性实现| 默认确定性实现|
| [aclnnGeScalar&aclnnInplaceGeScalar](aclnnGeScalar&aclnnInplaceGeScalar.md) | 判断输入Tensor中的每个元素是否大于等于other Scalar的值，返回一个Bool类型的Tensor，对应输入Tensor中每个位置的大于等于判断是否成立。 | 默认确定性实现| 默认确定性实现|
| [aclnnGeTensor&aclnnInplaceGeTensor](aclnnGeTensor&aclnnInplaceGeTensor.md) | 判断输入Tensor中的每个元素是否大于等于other Tensor的值，返回一个Bool类型的Tensor，对应输入Tensor中每个位置的大于等于判断是否成立。 | 默认确定性实现| 默认确定性实现 |
| [aclnnGlobalAveragePool](aclnnGlobalAveragePool.md) | 传入一个输入张量X，并在同一通道中的值上应用平均池化。        | 默认确定性实现| 默认确定性实现|
| [aclnnGlobalMaxPool](aclnnGlobalMaxPool.md) | 输入一个张量，并对同一通道中的值取最大值。                   | 默认确定性实现| 默认确定性实现|
| [aclnnGtScalar&aclnnInplaceGtScalar](aclnnGtScalar&aclnnInplaceGtScalar.md) | 判断输入Tensor中的每个元素是否大于other Scalar的值，返回一个Bool类型的Tensor，对应输入Tensor中每个位置的大于判断是否成立。 | 默认确定性实现| 默认确定性实现|
| [aclnnGtTensor&aclnnInplaceGtTensor](aclnnGtTensor&aclnnInplaceGtTensor.md)                                               | 判断self Tensor中的元素是否大于other Tensor中的元素。返回一个Tensor，若self>other，为True(1)；否则为False(0)。 | 默认确定性实现| 默认确定性实现|
| [aclnnGroupedBiasAddGrad](aclnnGroupedBiasAddGrad.md) | 分组偏置加法（GroupedBiasAdd）的反向传播。                   | 默认确定性实现| 默认确定性实现|
| [aclnnHansDecode](aclnnHansDecode.md) | 对压缩后的张量基于PDF进行解码，同时基于mantissa（尾数）重组恢复张量。 | 默认确定性实现| - |
| [aclnnHansEncode](aclnnHansEncode.md) | 对张量的指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩。 | 默认确定性实现| - |
| [aclnnHardtanh&aclnnInplaceHardtanh](aclnnHardtanh&aclnnInplaceHardtanh.md) | 将输入的所有元素限制在[clipValueMin,clipValueMax]范围内，若元素大于clipValueMax则限制为clipValueMax，若元素小于clipValueMin则限制为clipValueMin，否则等于元素本身。 | 默认确定性实现| 默认确定性实现|
| [aclnnHistc](aclnnHistc.md) | 以min和max作为统计上下限，在min和max之间划出等宽的数量为bins的区间，统计张量self中元素在各个区间的数量。如果min和max相等，则使用张量中所有元素的最小值和最大值作为统计的上下限。小于min和大于max的元素不会被统计。 | 默认确定性实现| 默认确定性实现|
| [aclnnIm2col](aclnnIm2col.md)      | 图像到列，滑动局部窗口数据转为列向量，拼接为大张量。从批处理输入张量中提取滑动窗口。 | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceAdds](aclnnInplaceAdds.md) | 完成加法计算。 | 默认确定性实现|  默认确定性实现 |
| [aclnnInplaceBitwiseAndScalar](aclnnInplaceBitwiseAndScalar.md) | 计算输入tensor中每个元素和输入标量other的按位与结果。输入的tensor与other必须是整型或者BOOL型变量。当输入为BOOL型时，计算逻辑与的结果。             | 默认确定性实现| 默认确定性实现 |
| [aclnnInplaceBitwiseAndTensor](aclnnInplaceBitwiseAndTensor.md) | 输入为BOOL型tensor时，进行逻辑与运算；输入为INT型时，进行位与运算。 | 默认确定性实现| 默认确定性实现 |
| [aclnnInplaceCopy](aclnnInplaceCopy.md) | 将src中的元素复制到selfRef张量中并返回selfRef。              | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceFillDiagonal](aclnnInplaceFillDiagonal.md) | 以fillValue填充tensor对角线。                                | 默认确定性实现| - |
| [aclnnInplaceFillScalar](aclnnInplaceFillScalar.md) | 对tensor填充指定标量。                                       | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceFillTensor](aclnnInplaceFillTensor.md) | 对selfRef张量填充value， value是张量。                       | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceMaskedFillScalar](aclnnInplaceMaskedFillScalar.md) | 用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素。 | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceMaskedFillTensor](aclnnInplaceMaskedFillTensor.md) | 用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素。 | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceNormalTensor](aclnnInplaceNormalTensor.md) | 使用mean和std的参数正态分布的元素进行填充张量。                | 默认确定性实现| 默认确定性实现  |
| [aclnnInplaceOne](aclnnInplaceOne.md) | 返回形状和类型相同的张量，所有元素都设置为1。                | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceRandomTensor](aclnnInplaceRandomTensor.md) | 从[from, to-1]的离散均匀分布中采样的数填充selfRef张量。                | 默认确定性实现| 默认确定性实现  |
| [aclnnInplaceUniformTensor](aclnnInplaceUniformTensor.md) | 生成[from, to)区间内离散均匀分布的随机数，并将其填充到selfRef张量中。                | 默认确定性实现| 默认确定性实现  |
| [aclnnInplaceZero](aclnnInplaceZero.md) | 将selfRef张量填充为全零。                                    | 默认确定性实现| 默认确定性实现|
| [aclnnIsClose](aclnnIsClose.md)        | 返回一个带有布尔元素的新张量，判断给定的self和other是否彼此接近，如果值接近，则返回True，否则返回False。 | 默认确定性实现| 默认确定性实现|
| [aclnnIsFinite](aclnnIsFinite.md)     | 判断张量中哪些元素是有限数值，即不是inf、-inf或nan。         | 默认确定性实现| 默认确定性实现|
| [aclnnIsInf](aclnnIsInf.md)              | 判断张量中哪些元素是无限大值，即是inf、-inf。                | 默认确定性实现| 默认确定性实现|
| [aclnnIsInScalarTensor](aclnnIsInScalarTensor.md) | 检查element中的元素是否等于testElement。                     | 默认确定性实现| 默认确定性实现|
| [aclnnIsInTensorScalar](aclnnIsInTensorScalar.md) | 检查element中的元素是否等于testElement。                     | 默认确定性实现| 默认确定性实现 |
| [aclnnIsNegInf](aclnnIsNegInf.md)    | 判断输入张量的元素是否为负无穷。                             | 默认确定性实现| 默认确定性实现|
| [aclnnIsPosInf](aclnnIsPosInf.md)    | 判断输入张量的元素是否为正无穷。                             | 默认确定性实现| 默认确定性实现|
| [aclnnKlDiv](aclnnKlDiv.md)           | 计算KL散度。                                                 | 默认确定性实现| 默认确定性实现|
| [aclnnLeftShift](aclnnLeftShift.md)           | 对于输入张量self中每个元素，根据输入张量shiftBits对应位置的参数，按位进行左移。           | 默认确定性实现| 默认确定性实现|
| [aclnnLeftShifts](aclnnLeftShifts.md)           | 对于输入张量self中每个元素，根据输入标量shiftBits，按位进行左移。   | 默认确定性实现| 默认确定性实现|
| [aclnnLerp&aclnnInplaceLerp](aclnnLerp&aclnnInplaceLerp.md) | 根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。 | 默认确定性实现| 默认确定性实现|
| [aclnnLerps&aclnnInplaceLerps](aclnnLerps&aclnnInplaceLerps.md) | 根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。 | 默认确定性实现| 默认确定性实现|
| [aclnnLeScalar&aclnnInplaceLeScalar](aclnnLeScalar&aclnnInplaceLeScalar.md) | 判断输入self中的元素值是否小于等于other的值，并将self的每个元素的值与other值的比较结果写入out中。 | 默认确定性实现| 默认确定性实现|
| [aclnnLeTensor&aclnnInplaceLeTensor](aclnnLeTensor&aclnnInplaceLeTensor.md) | 计算self中的元素值是否小于等于other的值，将self每个元素与other值的比较结果写入out中。 | 默认确定性实现| 默认确定性实现|
| [aclnnLinspace](aclnnLinspace.md)     | 生成一个等间隔数值序列。                                     | 默认确定性实现| 默认确定性实现|
| [aclnnLogSpace](aclnnLogSpace.md) | 创建一个大小为$\text{steps}$的一维张量，其值在$\text{base}^\text{start}$到$\text{base}^\text{end}$上对数尺度上均匀间隔，包含端点，以$\text{base}$为底。                                      | 默认确定性实现| - |
| [aclnnLog&aclnnInplaceLog](aclnnLog&aclnnInplaceLog.md) | 完成自然对数的计算                                           | 默认确定性实现| 默认确定性实现|
| [aclnnLog2&aclnnInplaceLog2](aclnnLog2&aclnnInplaceLog2.md) | 完成以2为底的对数计算                                        | 默认确定性实现| 默认确定性实现|
| [aclnnLog10&aclnnInplaceLog10](aclnnLog10&aclnnInplaceLog10.md) | 完成输入以10为底的对数计算                                   | 默认确定性实现| 默认确定性实现|
| [aclnnLog1p&aclnnInplaceLog1p](aclnnLog1p&aclnnInplaceLog1p.md) | 对输入Tensor完成log1p运算。                                  | 默认确定性实现| 默认确定性实现|
| [aclnnLogicalAnd&aclnnInplaceLogicalAnd](aclnnLogicalAnd&aclnnInplaceLogicalAnd.md) | 完成给定输入张量元素的逻辑与运算。0被视为False，非0被视为True。 | 默认确定性实现| 默认确定性实现|
| [aclnnLogicalNot&aclnnInplaceLogicalNot](aclnnLogicalNot&aclnnInplaceLogicalNot.md) | 计算给定输入Tensor的逐元素逻辑非。如果未指定输出类型，输出Tensor是bool类型。 | 默认确定性实现| 默认确定性实现|
| [aclnnLogicalOr&aclnnInplaceLogicalOr](aclnnLogicalOr&aclnnInplaceLogicalOr.md) | 完成给定输入张量元素的逻辑或运算。当两个输入张量为非bool类型时，0被视为False，非0被视为True。 | 默认确定性实现| 默认确定性实现|
| [aclnnLogicalXor](aclnnLogicalXor.md) | 完成给定输入张量元素的逻辑异或运算。当self和other为非bool类型时，0被视为False，非0被视为True。 | 默认确定性实现| 默认确定性实现 |
| [aclnnLogSumExp](aclnnLogSumExp.md) | 返回输入tensor指定维度上的指数之和的对数。                   | 默认确定性实现| 默认确定性实现|
| [aclnnLtScalar&aclnnInplaceLtScalar](aclnnLtScalar&aclnnInplaceLtScalar.md) | 判断输入self中的每个元素是否小于输入other的值，返回一个Bool类型的Tensor。 | 默认确定性实现| 默认确定性实现|
| [aclnnLtTensor&aclnnInplaceLtTensor](aclnnLtTensor&aclnnInplaceLtTensor.md) | 判断输入self中的每个元素是否小于输入other中的元素，返回一个Bool类型的Tensor。 | 默认确定性实现| 默认确定性实现|
| [aclnnMaskedScale](aclnnMaskedScale.md) | 完成elementwise计算。                                        | 默认确定性实现| 默认确定性实现|
| [aclnnMaskedSelect](aclnnMaskedSelect.md) | 根据一个布尔掩码张量（mask）中的值选择输入张量（self）中的元素作为输出，形成一个新的一维张量。 | 默认确定性实现| 默认确定性实现|
| [aclnnMaxDim](aclnnMaxDim.md) | 返回Tensor指定维度的最大值及其索引位置。                     | 默认确定性实现| 默认确定性实现|
| [aclnnMax](aclnnMax.md)              | 返回Tensor所有元素中的最大值。                               | 默认确定性实现| 默认确定性实现|
| [aclnnMaximum](aclnnMaximum.md)         | 计算两个张量中每个元素的最大值，并返回一个新的张量。         | 默认确定性实现| 默认确定性实现|
| [aclnnMaxN](aclnnMaxN.md)               | 对输入tensor列表中每个输入tensor对应元素求max。              | 默认确定性实现| 默认确定性实现|
| [aclnnMaxV2](aclnnMaxV2.md)          | 按指定维度对输入tensor求元素最大值。                         | 默认确定性实现| 默认确定性实现|
| [aclnnMean](aclnnMean.md)           | 按指定维度对Tensor求均值。                                   | 默认确定性实现| 默认确定性实现|
| [aclnnMeanV2](aclnnMeanV2.md)       | 按指定维度对Tensor求均值。                                   | 默认确定性实现| 默认确定性实现|
| [aclnnMin](aclnnMin.md)              | 返回Tensor所有元素中的最小值。                               | 默认确定性实现| 默认确定性实现|
| [aclnnMinDim](aclnnMinDim.md) | 返回self中指定维度的最小值及其索引位置。                     | 默认确定性实现| 默认确定性实现|
| [aclnnMinimum](aclnnMinimum.md)         | 计算两个张量中每个元素的最小值，并返回一个新的张量。         | 默认确定性实现| 默认确定性实现|
| [aclnnMinN](aclnnMinN.md)               | 对输入tensor列表中每个输入tensor对应元素求min。              | 默认确定性实现| 默认确定性实现|
| [aclnnMul&aclnnInplaceMul](aclnnMul&aclnnInplaceMul.md) | 完成tensor与tensor间的乘法计算。                             | 默认确定性实现| 默认确定性实现|
| [aclnnMuls&aclnnInplaceMuls](aclnnMuls&aclnnInplaceMuls.md) | 完成tensor与scalar间的乘法计算。                             | 默认确定性实现| 默认确定性实现|
| [aclnnNanToNum&aclnnInplaceNanToNum](aclnnNanToNum&aclnnInplaceNanToNum.md) | 将输入中的NaN、正无穷大和负无穷大值分别替换为nan、posinf、neginf指定的值。 | 默认确定性实现| 默认确定性实现|
| [aclnnNeg&aclnnInplaceNeg](aclnnNeg&aclnnInplaceNeg.md) | 对输入的每个元素完成相反数计算。                             | 默认确定性实现| 默认确定性实现|
| [aclnnNeScalar&aclnnInplaceNeScalar](aclnnNeScalar&aclnnInplaceNeScalar.md) | 计算selfRef中的元素的值与other的值是否不相等。               | 默认确定性实现| 默认确定性实现|
| [aclnnNeTensor&aclnnInplaceNeTensor](aclnnNeTensor&aclnnInplaceNeTensor.md) | 计算self（selfRef）中的元素的值与other的值是否不相等。       | 默认确定性实现| 默认确定性实现|
| [aclnnNormalFloatTensor](aclnnNormalFloatTensor.md)         | 返回一个随机数张量，该随机数是从给定的均值(float)和标准差(tensor)的独立正态分布中获取。 | 默认确定性实现| 默认确定性实现  |
| [aclnnNormalTensorFloat](aclnnNormalTensorFloat.md)         | 返回一个随机数张量，该随机数是从给定的均值(tensor)和标准差(float)的独立正态分布中获取。 | 默认确定性实现| 默认确定性实现  |
| [aclnnNormalTensorTensor](aclnnNormalTensorTensor.md)         | 返回一个随机数，该随机数是从给定的均值(tensor)和标准差(tensor)的独立正态分布中获取。 | 默认确定性实现| 默认确定性实现  |
| [aclnnNpuFormatCast](aclnnNpuFormatCast.md) | 完成ND←→NZ的转换功能。                                       | 默认确定性实现|   |
| [aclnnOneHot](aclnnOneHot.md)           | 对长度为n的输入self，经过one_hot的计算后得到一个元素数量为n*k的输出out。           | 默认确定性实现| 默认确定性实现|
| [aclnnPdist](aclnnPdist.md)               | 计算输入self中每对行向量的p范数距离。                        | 默认确定性实现| - |
| [aclnnPdistForward](aclnnPdistForward.md) | 计算输入self中每对行向量的p范数距离。                        | 默认确定性实现| - |
| [aclnnPermute](aclnnPermute.md) | 对tensor的任意维度进行调换。如输入self是shape为[2, 3, 5]的tensor，dims为(2, 0, 1)，则输出是shape为[5, 2, 3]的tensor。 | 默认确定性实现| 默认确定性实现|
| [aclnnPowScalarTensor](aclnnPowScalarTensor.md) | exponent每个元素作为input对应元素的幂完成计算。              | 默认确定性实现| 默认确定性实现|
| [aclnnPowTensorScalar&aclnnInplacePowTensorScalar](aclnnPowTensorScalar&aclnnInplacePowTensorScalar.md) | exponent每个元素作为input对应元素的幂完成计算。              | 默认确定性实现| 默认确定性实现|
| [aclnnPowTensorTensor&aclnnInplacePowTensorTensor](aclnnPowTensorTensor&aclnnInplacePowTensorTensor.md) | exponent每个元素作为input对应元素的幂完成计算。              | 默认确定性实现| 默认确定性实现|
| [aclnnProd](aclnnProd.md)           | 回输入tensor中所有元素的乘积。                               | 默认确定性实现| 默认确定性实现|
| [aclnnProdDim](aclnnProdDim.md)     | 返回输入tensor给定维度上每行的乘积。                         | 默认确定性实现| 默认确定性实现|
| [aclnnRange](aclnnRange.md)               | 从start起始到end结束按照step的间隔取值。                     | 默认确定性实现| 默认确定性实现|
| [aclnnReal](aclnnReal.md)                  | 为输入张量的每一个元素取实数部分。                           | 默认确定性实现| - |
| [aclnnReciprocal&aclnnInplaceReciprocal](aclnnReciprocal&aclnnInplaceReciprocal.md) | 返回一个具有每个输入元素倒数的新张量。                       | 默认确定性实现| 默认确定性实现|
| [aclnnReduceLogSum](aclnnReduceLogSum.md) | 返回给定维度中输入张量每行的和再取对数。                     | 默认确定性实现| - |
| [aclnnReduceNansum](aclnnReduceNansum.md) | 将tensor中NaN处理为0后，返回输入tensor给定维度上的和。       | 默认确定性实现| - |
| [aclnnReduceSum](aclnnReduceSum.md) | 返回给定维度中输入张量每行的和。                             | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad1d](aclnnReflectionPad1d.md) | 使用输入边界的反射填充输入tensor。                           | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad2d](aclnnReflectionPad2d.md) | 使用输入边界的反射填充输入tensor。                           | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad3d](aclnnReflectionPad3d.md) | 3D反射填充。                                                 | 默认确定性实现| 默认确定性实现|
| [aclnnRemainderScalarTensor](aclnnRemainderScalarTensor.md) | 将scalar self进行broadcast成和tensor other一样shape的tensor以后，其中的每个元素都转换为除以other的对应元素以后得到的余数。 | 默认确定性实现| 默认确定性实现|
| [aclnnRemainderTensorScalar&aclnnInplaceRemainderTensorScalar](aclnnRemainderTensorScalar&aclnnInplaceRemainderTensorScalar.md) | 将tensor self中的每个元素都转换为除以scalar other以后得到的余数。 | 默认确定性实现| 默认确定性实现|
| [aclnnRemainderTensorTensor&aclnnInplaceRemainderTensorTensor](aclnnRemainderTensorTensor&aclnnInplaceRemainderTensorTensor.md) | 将tensor self和tensor other进行broadcast成一致的shape后，其中的每个元素都转换为除以other的对应元素以后得到的余数。 | 默认确定性实现| 默认确定性实现|
| [aclnnRepeat](aclnnRepeat.md)              | 对输入tensor沿着repeats中对每个维度指定的复制次数进行复制。  | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad1d](aclnnReplicationPad1d.md) | 使用输入边界填充输入tensor的最后一维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad2d](aclnnReplicationPad2d.md) | 使用输入边界填充输入tensor的最后两维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad3d](aclnnReplicationPad3d.md) | 使用输入边界填充输入tensor的最后三维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad1d](aclnnReflectionPad1d.md) | 调用REFLECT情况下的mirror_pad算子，填充输入tensor的最后一维。 | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad2d](aclnnReflectionPad2d.md) | 调用REFLECT情况下的mirror_pad算子，填充输入tensor的最后两维。 | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad3d](aclnnReflectionPad3d.md) | 调用REFLECT情况下的mirror_pad算子，填充输入tensor的最后三维。 | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad1dBackward](aclnnReplicationPad1dBackward.md) | replication_pad1d的反向传播。                                | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad2dBackward](aclnnReplicationPad2dBackward.md) | replication_pad2d的反向传播, 前向计算参考aclnnReplicationPad2d。 | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad3dBackward](aclnnReplicationPad3dBackward.md) | 计算aclnnReplicationPad3d的反向传播。                        | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad1dBackward](aclnnReflectionPad1dBackward.md) | reflection_pad1d的反向传播，前向计算参考aclnnReflectionPad1d。 | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad2dBackward](aclnnReflectionPad2dBackward.md) | reflection_pad2d的反向传播，前向计算参考aclnnReflectionPad2d。 | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad3dBackward](aclnnReflectionPad3dBackward.md) | 计算aclnnReflectionPad3d api的反向传播。                     | 默认确定性实现| 默认确定性实现|
| [aclnnRightShift](aclnnRightShift.md)         | 对于输入张量input中每个元素，根据输入张量shiftBits对应位置的参数，按位进行右移。 | 默认确定性实现| 默认确定性实现|
| [aclnnRoll](aclnnRoll.md)            | 沿给定尺寸和维度移动Tensor中的数据。                         | 默认确定性实现| 默认确定性实现|
| [aclnnRound&aclnnInplaceRound](aclnnRound&aclnnInplaceRound.md) | 将输入的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。 | 默认确定性实现| 默认确定性实现|
| [aclnnRoundDecimals&aclnnInplaceRoundDecimals](aclnnRoundDecimals&aclnnInplaceRoundDecimals.md) | 将输入Tensor的元素四舍五入到指定的位数。                     | 默认确定性实现| 默认确定性实现|
| [aclnnRsqrt&aclnnInplaceRsqrt](aclnnRsqrt&aclnnInplaceRsqrt.md) | 求input(Tensor)每个元素的平方根的倒数。                      | 默认确定性实现| 默认确定性实现|
| [aclnnRsub](aclnnRsub.md)                   | 完成减法计算。                                               | 默认确定性实现| 默认确定性实现|
| [aclnnRsubs](aclnnRsubs.md)                   | 完成减法计算。                                               | 默认确定性实现| 默认确定性实现 |
| [aclnnScale](aclnnScale.md)               | 参见算子文档。                                               | 默认确定性实现| - |
| [aclnnSearchSorted](aclnnSearchSorted.md) | 在一个已排序的张量（sortedSequence）中查找给定tensor值（self）应该插入的位置。 | 默认确定性实现| - |
| [aclnnSign](aclnnSign.md)                  | 对输入的tensor逐元素进行Sign符号函数的运算并输出结果tensor。 | 默认确定性实现| 默认确定性实现|
| [aclnnSignbit](aclnnSignbit.md)            | 判断输入中的每个元素符号位是否为1，返回一个bool类型的tensor。 | 默认确定性实现| 默认确定性实现 |
| [aclnnSignBitsPack](aclnnSignBitsPack.md) | 将float16类型或者float32类型的1位Adam打包为uint8。           | 默认确定性实现| - |
| [aclnnSignBitsUnpack](aclnnSignBitsUnpack.md) | 将uint8类型1位Adam拆包为float32或者float16。                 | 默认确定性实现| - |
| [aclnnSilentCheck](aclnnSilentCheck.md) | SilentCheckV2算子功能主要根据输入特征值（val），与绝对阈值、相对阈值比较，来识别是否触发静默检测故障。 | 默认确定性实现| - |
| [aclnnSin&aclnnInplaceSin](aclnnSin&aclnnInplaceSin.md) | 对输入Tensor完成sin运算。                                    | 默认确定性实现| 默认确定性实现|
| [aclnnSinc&aclnnInplaceSinc](aclnnSinc&aclnnInplaceSinc.md) | 对输入Tensor完成sinc运算。                                   | 默认确定性实现| - |
| [aclnnSinh&aclnnInplaceSinh](aclnnSinh&aclnnInplaceSinh.md) | 对输入Tensor完成sinh运算。                                   | 默认确定性实现| - |
| [aclnnSinkhorn](aclnnSinkhorn.md)      | 计算Sinkhorn距离，可以用于MoE模型中的专家路由。              | 默认确定性实现| - |
| [aclnnSlice](aclnnSlice.md) | 在指定维度dim上，根据给定的范围[start,end]和步长step，从输入张量self中提取子张量out。 | 默认确定性实现| 默认确定性实现|
| [aclnnSliceV2](aclnnSliceV2.md) | 根据给定的维度axes、范围[starts,ends]和步长steps，从输入张量self中提取张量out。 | 默认确定性实现| 默认确定性实现|
| [aclnnSort](aclnnSort.md)                  | 将输入tensor中的元素根据指定维度进行升序/降序， 并且返回对应的index值。 | 默认确定性实现| 默认确定性实现 |
| [aclnnSplitTensor](aclnnSplitTensor.md) | 将输入self沿dim轴按照splitSections大小均匀切分。             | 默认确定性实现| 默认确定性实现|
| [aclnnSplitWithSize](aclnnSplitWithSize.md) | 将输入self沿dim轴切分至splitSize中每个元素的大小。           | 默认确定性实现| 默认确定性实现|
| [aclnnSqrt&aclnnInplaceSqrt](aclnnSqrt&aclnnInplaceSqrt.md) | 完成非负数平方根计算，负数情况返回nan。                      | 默认确定性实现| 默认确定性实现|
| [aclnnSquare](aclnnSquare.md) | 为输入张量的每一个元素计算平方值。                      | 默认确定性实现| 默认确定性实现|
| [aclnnStack](aclnnStack.md)          | 沿着新维度连接张量序列。                                     | 默认确定性实现| 默认确定性实现  |
| [aclnnStd](aclnnStd.md)           | 计算指定维度(dim)的标准差，这个dim可以是单个维度，维度列表或者None。 | 默认确定性实现| 默认确定性实现|
| [aclnnStdMeanCorrection](aclnnStdMeanCorrection.md) | 计算样本标准差和均值。                                       | 默认确定性实现| 默认确定性实现|
| [aclnnStridedSliceAssignV2](aclnnStridedSliceAssignV2.md) | StridedSliceAssign是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。 | 默认确定性实现| 默认确定性实现|
| [aclnnSub&aclnnInplaceSub](aclnnSub&aclnnInplaceSub.md) | 完成减法计算，被减数按alpha进行缩放。                        | 默认确定性实现| 默认确定性实现|
| [aclnnSubs&aclnnInplaceSubs](aclnnSubs&aclnnInplaceSubs.md) | 完成减法计算，被减数按alpha进行缩放。                        | 默认确定性实现| 默认确定性实现 |
| [aclnnSum](aclnnSum.md)          | 返回输入tensors列表中每个输入tensor依次做add求和。           | 默认确定性实现| - |
| [aclnnSWhere](aclnnSWhere.md)            | 根据条件选取self或other中元素并返回（支持广播）。            | 默认确定性实现| 默认确定性实现|
| [aclnnTan&aclnnInplaceTan](aclnnTan&aclnnInplaceTan.md) | 它计算输入张量self中每个元素的正切值，并将结果存储在输出张量out中。 | 默认确定性实现| - |
| [aclnnTanh&aclnnInplaceTanh](aclnnTanh&aclnnInplaceTanh.md) | 激活函数。返回与输入tensor shape相同的tensor，对输入tensor进行elementwise的计算。 | 默认确定性实现| 默认确定性实现|
| [aclnnTanhBackward](aclnnTanhBackward.md) | aclnnTanh的反向。                                            | 默认确定性实现| 默认确定性实现|
| [aclnnTopk](aclnnTopk.md) | 返回输入Tensor在指定维度上的k个极值及索引。                                            | 默认确定性实现| 默认确定性实现 |
| [aclnnTransformBiasRescaleQkv](aclnnTransformBiasRescaleQkv.md) | TransformBiasRescaleQkv 算子是一个用于处理多头注意力机制中查询（Query）、键（Key）、值（Value）向量的接口。 | 默认确定性实现|   |
| [aclnnTransMatmulWeight](aclnnTransMatmulWeight.md) | 需要和aclnnCalculateMatmulWeightSize、aclnnCalculateMatmulWeightSizeV2接口配套使用，用于创建一个对于Matmul算子计算性能亲和的weight Tensor。 | 默认确定性实现|   |
| [aclnnTriangularSolve](aclnnTriangularSolve.md) | 求解一个具有方形上或下三角形可逆矩阵A和多个右侧b的方程组。   | 默认确定性实现| 默认确定性实现|
| [aclnnTril&aclnnInplaceTril](aclnnTril&aclnnInplaceTril.md) | 将输入的self张量的最后二维（按shape从左向右数）沿对角线的右上部分置零。   | 默认确定性实现| 默认确定性实现  |
| [aclnnTriu&aclnnInplaceTriu](aclnnTriu&aclnnInplaceTriu.md) | 将输入的self张量的最后二维（按shape从左向右数）沿对角线的左下部分置零。   | 默认确定性实现|  默认确定性实现 |
| [aclnnTrunc&aclnnInplaceTrunc](aclnnTrunc&aclnnInplaceTrunc.md) | 对输入Tensor完成trunc运算（将数字的小数部分截去，返回整数部分）。 | 默认确定性实现| 默认确定性实现|
| [aclnnUnfoldGrad](aclnnUnfoldGrad.md) | 实现Unfold算子的反向功能，计算相应的梯度。                   | 默认确定性实现|   |
| [aclnnVar](aclnnVar.md)              | 返回输入Tensor指定维度的值求得的方差。                       | 默认确定性实现| 默认确定性实现|
| [aclnnVarMean](aclnnVarMean.md)      | 返回输入Tensor指定维度的值求得的均值及方差。                 | 默认确定性实现| 默认确定性实现|
| [aclnnXLogYScalarOther&aclnnInplaceXLogYScalarOther](aclnnXLogYScalarOther&aclnnInplaceXLogYScalarOther.md) | 计算self * log(other)的结果。                                | 默认确定性实现| - |
| [aclnnXLogYScalarSelf](aclnnXLogYScalarSelf.md) | 计算self * log(other)的结果。                                | 默认确定性实现| - |
| [aclnnXLogYTensor&aclnnInplaceXLogYTensor](aclnnXLogYTensor&aclnnInplaceXLogYTensor.md) | 计算self * log(other)的结果。                                | 默认确定性实现| - |
| [aclRfft1D](aclRfft1D.md)               | 对输入张量self进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。 | 默认确定性实现| - |
| [aclStft](aclStft.md)                      | 计算输入在滑动窗口内的傅里叶变换。                           | 默认确定性实现| - |
