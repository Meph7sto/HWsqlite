# 算子接口（aclnn）

## 使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在```${INSTALL_DIR}/include/aclnnop```，库文件默认在```${INSTALL_DIR}/lib64```，具体文件如下：

- 依赖的头文件：①方式1 （推荐）：引用算子总头文件aclnn\_ops\_\$\{ops\_project\}.h。②方式2：按需引用单算子API头文件aclnn\_\*.h。
- 依赖的库文件：按需引用算子总库文件libopapi\_\$\{ops\_project\}.so。

其中${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓（如math、nn、cv、transformer），请配置为实际算子仓名。

## 接口列表

> **确定性简介**：
>
> - 配置说明：因CANN或NPU型号不同等原因，可能无法保证同一个算子多次运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分算子接口可通过`aclrtCtxSetSysParamOpt`（参见[《Runtime运行时 API》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/aolapi/operatorlist_00001.html)）开启确定性算法，使多次运行结果一致。
> - 性能说明：同一个算子采用确定性计算通常比非确定性慢，因此模型单次运行性能可能会下降。但在实验、调试和调测等需要保证多次运行结果相同来定位问题的场景，确定性计算可以提升效率。
> - 线程说明：同一线程中只能设置一次确定性状态，多次设置以最后一次有效设置为准。有效设置是指设置确定性状态后，真正执行了一次算子任务下发。如果仅设置，没有算子下发，只能是确定性变量开启但未下发给算子，因此不执行算子。
>   解决方案：暂不推荐一个线程多次设置确定性。该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。

算子接口列表如下：

|    接口名   |   说明     | 确定性说明（A2/A3） | 确定性说明（Atlas 350 加速卡） |
| ----------- | ------------------- | --------- |------------- |
|[aclnnAllGatherMatmul](aclnnAllGatherMatmul.md)|完成AllGather通信与MatMul计算融合。|默认确定性实现|默认确定性实现|
|[aclnnAllGatherMatmulV2](aclnnAllGatherMatmulV2.md)|aclnnAllGatherMatmulV2接口是对aclnnAllGatherMatmul接口的功能拓展。|默认确定性实现|默认确定性实现|
|[aclnnAlltoAllAllGatherBatchMatMul](aclnnAlltoAllAllGatherBatchMatMul.md)|完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。|默认确定性实现|-|
|[aclnnAlltoAllMatmul](aclnnAlltoAllMatmul.md)|完成AlltoAll通信与MatMul计算融合。|默认确定性实现|默认确定性实现|
|[aclnnAlltoAllQuantMatmul](aclnnAlltoAllQuantMatmul.md)|完成AlltoAll通信、量化计算、MatMul计算和反量化计算的融合。|默认确定性实现|默认确定性实现|
|[aclnnAlltoAllvGroupedMatMul](aclnnAlltoAllvGroupedMatMul.md)|完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合。|默认确定性实现|默认确定性实现|
|[aclnnApplyRotaryPosEmb](aclnnApplyRotaryPosEmb.md)|将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。|默认确定性实现|默认确定性实现|
|[aclnnApplyRotaryPosEmbV2](aclnnApplyRotaryPosEmbV2.md)|将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。|默认确定性实现|默认确定性实现|
|[aclnnAttentionUpdate](aclnnAttentionUpdate.md)|将各SP域PA算子的输出的中间结果lse，localOut两个局部变量结果更新成全局结果。|默认确定性实现|默认确定性实现|
|[aclnnBatchMatMulReduceScatterAlltoAll](aclnnBatchMatMulReduceScatterAlltoAll.md)|BatchMatMulReduceScatterAllToAll是通算融合算子，实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行的算子。|默认确定性实现|默认确定性实现|
|[aclnnBlockSparseAttentionGrad](aclnnBlockSparseAttentionGrad.md)|BlockSparseAttentionGrad通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。|默认确定性实现|默认确定性实现|
|[aclnnFusedCausalConv1d](aclnnFusedCausalConv1d.md)|对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列尾部的数据（长度为卷积核宽减1）更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。|默认确定性实现|默认确定性实现|
|[aclnnAttentionToFFN](aclnnAttentionToFFN.md)|将Attention节点上数据发往FFN节点。|默认确定性实现|默认确定性实现|
|[aclnnFFNToAttention](aclnnFFNToAttention.md)|将FFN节点上的token数据发往Attention节点。|默认非确定性实现|默认确定性实现|
|[aclnnDequantRopeQuantKvcache](aclnnDequantRopeQuantKvcache.md)|对输入张量进行dequant后，对尾轴进行切分，划分为q、k、vOut，对q、k进行旋转位置编码，并进行量化。|默认确定性实现|默认确定性实现|
|[aclnnDistributeBarrier](aclnnDistributeBarrier.md)|完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。|默认确定性实现|默认确定性实现|
|[aclnnDistributeBarrierV2](aclnnDistributeBarrierV2.md)|完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。|默认确定性实现|默认确定性实现|
|[aclnnDenseLightningIndexerGradKLLoss](aclnnDenseLightningIndexerGradKLLoss.md)|dense场景LightningIndexer的反向算子，再额外融合了Loss计算功能。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnDenseLightningIndexerSoftmaxLse](aclnnDenseLightningIndexerSoftmaxLse.md)|dense场景DenseLightningIndexerGradKlLoss算子计算Softmax输入的一个分支算子。|默认确定性实现|默认确定性实现|
|[aclnnFFN](aclnnFFN.md)|该FFN算子提供MoeFFN和FFN的计算功能。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFFNV2](aclnnFFNV2.md)|该FFN算子提供MoeFFN和FFN的计算功能。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFFNV3](aclnnFFNV3.md)|该FFN算子提供MoeFFN和FFN的计算功能。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionScore](aclnnFlashAttentionScore.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现|默认确定性实现|
|[aclnnFlashAttentionScoreV2](aclnnFlashAttentionScoreV2.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现|默认确定性实现|
|[aclnnFlashAttentionScoreV3](aclnnFlashAttentionScoreV3.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。对标竞品适配gptoss模型支持sink功能。|默认确定性实现|默认确定性实现|
|[aclnnFlashAttentionScoreGrad](aclnnFlashAttentionScoreGrad.md)|训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionScoreGradV2](aclnnFlashAttentionScoreGradV2.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionScoreV2](aclnnFlashAttentionScoreV2.md)的反向计算。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionScoreGradV3](aclnnFlashAttentionScoreGradV3.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionScoreV3](aclnnFlashAttentionScoreV3.md)的反向计算。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionScoreGradV4](aclnnFlashAttentionScoreGradV4.md)|训练场景下计算注意力的反向输出，即[FlashAttentionScoreV4](aclnnFlashAttentionScoreV4.md)的反向计算。该接口query、key、value参数支持多个长度相等或者长度不相等的sequence。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionUnpaddingScoreGrad](aclnnFlashAttentionUnpaddingScoreGrad.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionVarLenScore](aclnnFlashAttentionVarLenScore.md)的反向计算。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionUnpaddingScoreGradV2](aclnnFlashAttentionUnpaddingScoreGradV2.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionVarLenScoreV2](aclnnFlashAttentionVarLenScoreV2.md)的反向计算。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionUnpaddingScoreGradV3](aclnnFlashAttentionUnpaddingScoreGradV3.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionVarLenScoreV3](aclnnFlashAttentionVarLenScoreV3.md)的反向计算。该接口相较于[aclnnFlashAttentionUnpaddingScoreGradV2](aclnnFlashAttentionUnpaddingScoreGradV2.md)接口，新增queryRope、keyRope、dqRope和dkRope参数。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionUnpaddingScoreGradV4](aclnnFlashAttentionUnpaddingScoreGradV4.md)|训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionUnpaddingScoreGradV5](aclnnFlashAttentionUnpaddingScoreGradV5.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。增加`sinkInOptional`可选输入。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnFlashAttentionVarLenScore](aclnnFlashAttentionVarLenScore.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现|默认确定性实现|
|[aclnnFlashAttentionVarLenScoreV2](aclnnFlashAttentionVarLenScoreV2.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现|默认确定性实现|
|[aclnnFlashAttentionVarLenScoreV3](aclnnFlashAttentionVarLenScoreV3.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现|默认确定性实现|
|[aclnnFlashAttentionVarLenScoreV4](aclnnFlashAttentionVarLenScoreV4.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现|默认确定性实现|
|[aclnnFlashAttentionVarLenScoreV5](aclnnFlashAttentionVarLenScoreV5.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。对标竞品适配gptoss模型支持sink功能。|默认确定性实现|默认确定性实现|
|[aclnnFusedInferAttentionScoreV4](aclnnFusedInferAttentionScoreV4.md)|适配Decode & Prefill场景的FlashAttention算子。|默认确定性实现|默认确定性实现|
|[aclnnFusedInferAttentionScoreV5](aclnnFusedInferAttentionScoreV5.md)|适配增量&全量推理场景的FlashAttention算子。|默认确定性实现|默认确定性实现|
|[aclnnGatherPaKvCache](aclnnGatherPaKvCache.md)|根据blockTables中的blockId值、seqLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。|��认��定性实现|默认确定性实现|
|[aclnnGroupedMatmulV5](aclnnGroupedMatmulV5.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulAdd](aclnnGroupedMatmulAdd.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatMulAlltoAllv](aclnnGroupedMatMulAlltoAllv.md)|完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulFinalizeRouting](aclnnGroupedMatmulFinalizeRouting.md)|GroupedMatMul和MoeFinalizeRouting的融合算子。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnGroupedMatmulFinalizeRoutingV2](aclnnGroupedMatmulFinalizeRoutingV2.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnGroupedMatmulFinalizeRoutingV3](aclnnGroupedMatmulFinalizeRoutingV3.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。|默认非确定性实现，支持配置开启|默认非确定性实现，不支持配置开启|
|[aclnnGroupedMatmulFinalizeRoutingWeightNz](aclnnGroupedMatmulFinalizeRoutingWeightNz.md)|GroupedMatMul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持输入Weight为AI处理器亲和数据排布格式(NZ)。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnGroupedMatmulFinalizeRoutingWeightNzV2](aclnnGroupedMatmulFinalizeRoutingWeightNzV2.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持w为AI处理器亲和数据排布格式(NZ)。|默认非确定性实现，支持配置开启|默认非确定性实现，不支持配置开启|
|[aclnnGroupedMatmulSwigluQuant](aclnnGroupedMatmulSwigluQuant.md)|融合GroupedMatMul、Dequant、Swiglu和Quant。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulSwigluQuantV2](aclnnGroupedMatmulSwigluQuantV2.md)|融合GroupedMatmul 、dequant、swiglu和quant。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulSwigluQuantWeightNZ](aclnnGroupedMatmulSwigluQuantWeightNZ.md)|融合GroupedMatMul、Dequant、Swiglu和Quant，输入权重Weight会被强制视为NZ格式。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulWeightNz](aclnnGroupedMatmulWeightNz.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同，输入权重Weight会被强制视为NZ格式。|默认确定性实现|默认确定性实现|
|[aclnnIncreFlashAttentionV4](aclnnIncreFlashAttentionV4.md)|在全量推理场景的FlashAttention算子的基础上实现增量推理。|默认确定性实现|默认确定性实现|
|[aclnnInplaceAttentionWorkerScheduler](aclnnInplaceAttentionWorkerScheduler.md)|Attention和FFN分离部署场景下，Attention侧数据扫描算子。该算子接收来自FFNToAttention算子的输出数据，并对数据进行逐步扫描，确保数据准备就绪。|默认确定性实现|默认确定性实现|
|[aclnnInplaceFfnWorkerScheduler](aclnnInplaceFfnWorkerScheduler.md)|Attention和FFN分离场景下，FFN侧数据扫描算子。该算子接收AttentionToFFN算子发送的数据，进行扫描并完成数据整理。|默认确定性实现|默认确定性实现|
|[aclnnInterleaveRope](aclnnInterleaveRope.md)|针对单输入 x 进行旋转位置编码。|默认确定性实现|默认确定性实现|
|[aclnnLightningIndexerGrad](aclnnLightningIndexerGrad.md)|训练场景下，实现LightningIndexer反向，其中输入有Query, Key, Weights, Dy, Indices，反向主要利用正向计算的Indices从Key中提取TopK序列从而降低Matmul计算量。|默认非确定性实现，不支持配置开启|默认非确定性实现，不支持配置开启|
|[aclnnMatmulAlltoAll](aclnnMatmulAlltoAll.md)|完成MatMul计算与AlltoAll通信融合。|默认确定性实现|默认确定性实现|
|[aclnnMatmulAllReduce](aclnnMatmulAllReduce.md)|完成MatMul计算与AllReduce通信融合。|默认非确定性实现，支持配置开启|默认确定性实现|
|[aclnnMatmulAllReduceV2](aclnnMatmulAllReduceV2.md)|完成MatMul计算与AllReduce通信融合。|默认非确定性实现，支持配置开启|默认确定性实现|
|[aclnnMatmulReduceScatter](aclnnMatmulReduceScatter.md)|完成mm + reduce_scatter_base计算。|默认非确定性实现，支持配置开启|默认确定性实现|
|[aclnnMatmulReduceScatterV2](aclnnMatmulReduceScatterV2.md)|aclnnMatmulReduceScatterV2接口是对[aclnnMatmulReduceScatter](aclnnMatmulReduceScatter.md)接口的功能扩展。|默认非确定性实现，支持配置开启|默认确定性实现|
|[aclnnMlaPreprocess](aclnnMlaPreprocess.md)|Multi-Head Latent Attention前处理的计算 。|默认确定性实现|默认确定性实现|
|[aclnnMlaPreprocessV2](aclnnMlaPreprocessV2.md)|推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程如下：|默认确定性实现|默认确定性实现|
|[aclnnMlaProlog](aclnnMlaProlog.md)|Multi-Head Latent Attention前处理的计算 。|默认确定性实现|默认确定性实现|
|[aclnnMlaPrologV2WeightNz](aclnnMlaPrologV2WeightNz.md)|Multi-Head Latent Attention前处理的计算 。|默认确定性实现|默认确定性实现|
|[aclnnMlaPrologV3WeightNz](aclnnMlaPrologV3WeightNz.md)|Multi-Head Latent Attention前处理的计算 。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnMoeComputeExpertTokens](aclnnMoeComputeExpertTokens.md)|MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeCombine](aclnnMoeDistributeCombine.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeCombineV2](aclnnMoeDistributeCombineV2.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeCombineV3](aclnnMoeDistributeCombineV3.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeCombineV4](aclnnMoeDistributeCombineV4.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeCombineAddRmsNorm](aclnnMoeDistributeCombineAddRmsNorm.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合，之后完成Add + RmsNorm融合。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeCombineAddRmsNormV2](aclnnMoeDistributeCombineAddRmsNormV2.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合，之后完成Add + RmsNorm融合。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeDispatch](aclnnMoeDistributeDispatch.md)|对Token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeDispatchV2](aclnnMoeDistributeDispatchV2.md)|对token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeDispatchV3](aclnnMoeDistributeDispatchV3.md)|对token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现|默认确定性实现|
|[aclnnMoeDistributeDispatchV4](aclnnMoeDistributeDispatchV4.md)|对token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现|默认确定性实现|
|[aclnnMoeFinalizeRouting](aclnnMoeFinalizeRouting.md)|MoE计算中，最后处理合并MoE FFN的输出结果。|默认确定性实现|默认确定性实现|
|[aclnnMoeFinalizeRoutingV2](aclnnMoeFinalizeRoutingV2.md)|MoE计算中，最后处理合并MoE FFN的输出结果，支持配置dropPadMode。|默认确定性实现|默认确定性实现|
|[aclnnMoeFinalizeRoutingV2Grad](aclnnMoeFinalizeRoutingV2Grad.md)|aclnnMoeFinalizeRoutingV2的反向传播。|默认确定性实现|默认确定性实现|
|[aclnnMoeFusedTopk](aclnnMoeFusedTopk.md)|MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。|默认确定性实现|默认确定性实现|
|[aclnnMoeGatingTopK](aclnnMoeGatingTopK.md)|MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。|默认确定性实现|默认确定性实现|
|[aclnnMoeGatingTopKSoftmax](aclnnMoeGatingTopKSoftmax.md)|MoE计算中，对x的输出做Softmax计算，取TopK操作。|默认确定性实现|默认确定性实现|
|[aclnnMoeGatingTopKSoftmaxV2](aclnnMoeGatingTopKSoftmaxV2.md)|MoE��算中，如果renorm=0，先对x的输出做Softmax计算，再取TopK操作；如果renorm=1，先对x的输出做TopK操作，再进行Softmax操作。|默认确定性实现|默认确定性实现|
|[aclnnMoeInitRouting](aclnnMoeInitRouting.md)|MoE的routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做Routing处理。|默认确定性实现|默认确定性实现|
|[aclnnMoeInitRoutingV2](aclnnMoeInitRoutingV2.md)|该算子对应MoE中的Routing计算，以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用。|默认确定性实现|默认确定性实现|
|[aclnnMoeInitRoutingV3](aclnnMoeInitRoutingV3.md)|MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理，支持不量化和动态量化模式。|默认确定性实现|默认确定性实现|
|[aclnnMoeInitRoutingQuant](aclnnMoeInitRoutingQuant.md)|MoE的Routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做Routing处理，并对结果进行量化。|默认确定性实现|默认确定性实现|
|[aclnnMoeInitRoutingQuantV2](aclnnMoeInitRoutingQuantV2.md)|MoE的Routing计算，根据aclnnMoeGatingTopKSoftmaxV2的计算结果做Routing处理。|默认确定性实现|默认确定性实现|
|[aclnnMoeInitRoutingV2Grad](aclnnMoeInitRoutingV2Grad.md)|[aclnnMoeInitRoutingV2](aclnnMoeInitRoutingV2.md)的反向传播，完成Tokens的加权求和。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenPermute](aclnnMoeTokenPermute.md)|MoE的permute计算，根据索引indices将tokens广播并排序。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenPermuteGrad](aclnnMoeTokenPermuteGrad.md)|[aclnnMoeTokenPermute](aclnnMoeTokenPermute.md)的反向传播计算。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenPermuteWithEp](aclnnMoeTokenPermuteWithEp.md)|MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenPermuteWithEpGrad](aclnnMoeTokenPermuteWithEpGrad.md)|[aclnnMoeTokenPermuteWithEp](aclnnMoeTokenPermuteWithEp.md)的反向传播计算。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenPermuteWithRoutingMap](aclnnMoeTokenPermuteWithRoutingMap.md)|MoE的permute计算，将token和expert的标签作为routingMap传入，根据routingMaps将tokens和可选probsOptional广播后排序|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenPermuteWithRoutingMapGrad](aclnnMoeTokenPermuteWithRoutingMapGrad.md)|[aclnnMoeTokenPermuteWithRoutingMap](aclnnMoeTokenPermuteWithRoutingMap.md)的反向传播。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenUnpermute](aclnnMoeTokenUnpermute.md)|根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘；最后进行累加求和，并输出计算结果。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenUnpermuteGrad](aclnnMoeTokenUnpermuteGrad.md)|[aclnnMoeTokenUnpermute](aclnnMoeTokenUnpermute.md)的反向传播。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenUnpermuteWithEp](aclnnMoeTokenUnpermuteWithEp.md)|根据sortedIndices存储的下标位置，去获取permutedTokens中的输入数据与probs相乘，并进行合并累加。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenUnpermuteWithEpGrad](aclnnMoeTokenUnpermuteWithEpGrad.md)|[aclnnMoeTokenUnpermuteWithEp](aclnnMoeTokenUnpermuteWithEp.md)的反向传播。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenUnpermuteWithRoutingMap](aclnnMoeTokenUnpermuteWithRoutingMap.md)|对经过aclnnMoeTokenpermuteWithRoutingMap处理的permutedTokens，累加回原unpermutedTokens。|默认确定性实现|默认确定性实现|
|[aclnnMoeTokenUnpermuteWithRoutingMapGrad](aclnnMoeTokenUnpermuteWithRoutingMapGrad.md)|[aclnnMoeTokenUnpermuteWithRoutingMap](aclnnMoeTokenUnpermuteWithRoutingMap.md)的反向传播。|默认确定性实现|默认确定性实现|
|[aclnnMoeUpdateExpert](aclnnMoeUpdateExpert.md)|本API支持负载均衡和专家剪枝功能。经过映射后的专家表和Mask可传入MoE层进行数据分发和处理。|默认确定性实现|默认确定性实现|
|[aclnnMhcPre](aclnnMhcPre.md)|基于一系列计算得到MHC架构中hidden层的$H^{res}$和$H^{post}$投影矩阵以及Attention或MLP层的输入矩阵$h^{in}$。|默认确定性实现|默认确定性实现|
|[aclnnMhcPost](aclnnMhcPost.md)|基于一系列计算对mHC架构中上一层输出进行Post Mapping，对上一层的输入进行Res Mapping，然后对二者进行残差连接，得到下一层的输入。|默认确定性实现|默认确定性实现|
|[aclnnNormRopeConcat](aclnnNormRopeConcat.md)|transfomer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）。|默认确定性实现|默认确定性实现|
|[aclnnNormRopeConcatBackward](aclnnNormRopeConcatBackward.md)|transfomer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）融合算子功能反向推导。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnNsaCompress](aclnnNsaCompress.md)|训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。|默认确定性实现|默认确定性实现|
|[aclnnNsaCompressAttention](aclnnNsaCompressAttention.md)|NSA中compress attention以及select topk索引计算。|默认确定性实现|默认确定性实现|
|[aclnnNsaCompressAttentionInfer](aclnnNsaCompressAttentionInfer.md)|Native Sparse Attention推理过程中，Compress Attention的计算。|默认确定性实现|默认确定性实现|
|[aclnnNsaCompressGrad](aclnnNsaCompressGrad.md)|[aclnnNsaCompress](aclnnNsaCompress.md)算子的反向计算。|默认确定性实现|默认确定性实现|
|[aclnnNsaCompressWithCache](aclnnNsaCompressWithCache.md)|实现Native-Sparse-Attention推理阶段的KV压缩。|默认确定性实现|默认确定性实现|
|[aclnnNsaSelectedAttention](aclnnNsaSelectedAttention.md)|训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算。|默认确定性实现|默认确定性实现|
|[aclnnNsaSelectedAttentionGrad](aclnnNsaSelectedAttentionGrad.md)|���据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启|默认非确定性实现，支持配置开启|
|[aclnnNsaSelectedAttentionInfer](aclnnNsaSelectedAttentionInfer.md)|Native Sparse Attention推理过程中，Selected Attention的计算。|默认确定性实现|默认确定性实现|
|[aclnnPromptFlashAttentionV3](aclnnPromptFlashAttentionV3.md)|全量推理场景的FlashAttention算子。|默认确定性实现|默认确定性实现|
|[aclnnQkvRmsNormRopeCache](aclnnQkvRmsNormRopeCache.md)|输入qkv融合张量，通过SplitVD拆分q、k、v张量，执行RmsNorm、ApplyRotaryPosEmb、Quant、Scatter融合操作，输出qOut、kCache、vCache、qBeforeQuant(可选)、kBeforeQuant(可选)、vBeforeQuant(可选)。|默认确定性实现|默认确定性实现|
|[aclnnQuantGroupedMatmulDequantWeightNZ](aclnnQuantGroupedMatmulDequantWeightNZ.md)|对输入x进行量化，分组矩阵乘以及反量化，输入权重Weight会被强制视为NZ格式。|默认确定性实现|默认确定性实现|
|[aclnnQuantMatmulAllReduce](aclnnQuantMatmulAllReduce.md)|对量化后的入参x1、x2进行MatMul计算后，接着进行Dequant计算，接着与x3进行Add操作，最后做AllReduce计算。|默认非确定性实现，支持配置开启|默认确定性实现|
|[aclnnQuantMatmulAllReduceV2](aclnnQuantMatmulAllReduceV2.md)|aclnnQuantMatmulAllReduceV2接口是对[aclnnQuantMatmulAllReduce](aclnnQuantMatmulAllReduce.md)接口的功能扩展。|默认非确定性实现，支持配置开启|默认确定性实现|
|[aclnnQuantMatmulAllReduceV3](aclnnQuantMatmulAllReduceV3.md)|aclnnQuantMatmulAllReduceV3接口是对[aclnnQuantMatmulAllReduceV2](aclnnQuantMatmulAllReduceV2.md)接口的功能扩展。|默认非确定性实现，支持配置开启|默认确定性实现|
|[aclnnQuantMatmulAllReduceV4](aclnnQuantMatmulAllReduceV4.md)|兼容[aclnnQuantMatmulAllReduceV3](aclnnQuantMatmulAllReduceV3.md)支持的功能，在此基础上新增perblock量化方式的支持。|默认非确定性实现，支持配置开启|默认确定性实现|
|[aclnnQuantMatmulAlltoAll](aclnnQuantMatmulAlltoAll.md)|对量化后的入参x1、x2进行MatMul计算后，接着进行Dequant计算，最后做AlltoAll通信。|默认确定性实现|默认确定性实现|
|[aclnnQuantReduceScatter](aclnnQuantReduceScatter.md)|实现quant + reduceScatter融合计算。|默认确定性实现|默认确定性实现|
|[aclnnRainFusionAttention](aclnnRainFusionAttention.md)|RainFusionAttention稀疏注意力计算，支持灵活的块级稀疏模式，通过selectIdx指定每个Q块选择的KV块，实现高效的稀疏注意力计算。|默认确定性实现|默认确定性实现|
|[aclnnRecurrentGatedDeltaRule](aclnnRecurrentGatedDeltaRule.md)|完成变步长的Recurrent Gated Delta Rule计算。|默认确定性实现|默认确定性实现|
|[aclnnRingAttentionUpdate](aclnnRingAttentionUpdate.md)|将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|默认确定性实现|默认确定性实现|
|[aclnnRingAttentionUpdateV2](aclnnRingAttentionUpdateV2.md)|指定softmax的输入排布，将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|默认确定性实现|默认确定性实现|
|[aclnnRopeWithSinCosCache](aclnnRopeWithSinCosCache.md)|推理网络为了提升性能，将sin和cos输入通过cache传入，执行旋转位置编码计算。|默认确定性实现|默认确定性实现|
|[aclnnRopeWithSinCosCacheV2](aclnnRopeWithSinCosCacheV2.md)|对比V1增加cacheMode属性，指示cos和sin的拼接方式。|默认确定性实现|默认确定性实现|
|[aclnnRotaryPositionEmbedding](aclnnRotaryPositionEmbedding.md)|执行单路旋转位置编码计算。|默认确定性实现|默认确定性实现|
|[aclnnRotaryPositionEmbeddingGrad](aclnnRotaryPositionEmbeddingGrad.md)|单路旋转位置编码[aclnnRotaryPositionEmbedding](aclnnRotaryPositionEmbedding.md)的反向计算。|默认确定性实现|默认确定性实现|
|[aclnnScatterPaCache](aclnnScatterPaCache.md)|更新KCache中指定位置的key。|默认确定性实现|默认确定性实现|
|[aclnnScatterPaKvCache](aclnnScatterPaKvCache.md)|更新KvCache中指定位置的key和value。|默认确定性实现|默认确定性实现|
|[aclnnSparseFlashAttentionGrad](aclnnSparseFlashAttentionGrad.md)|根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启|-|
|[aclnnSparseLightningIndexerGradKLLoss](aclnnSparseLightningIndexerGradKLLoss.md)|LightningIndexer的反向算子，再额外融合了Loss计算功能。|默认非确定性实现，不支持配置开启|-|
|[aclnnWeightQuantMatmulAllReduce](aclnnWeightQuantMatmulAllReduce.md)|对入参x2进行伪量化计算后，完成MatMul和AllReduce计算。|默认非确定性实现，支持配置开启|默认确定性实现|
|[aclnnFusedFloydAttention](aclnnFusedFloydAttention.md)|训练场景下，使用FloydAttention算法实现多维自注意力的计算。|默认确定性实现| - |
|[aclnnFusedFloydAttentionGrad](aclnnFusedFloydAttentionGrad.md)|训练场景下，计算Floyd注意力的反向输出，FloydAttn相较于传统FA主要是计算qk/pv注意力时会额外将seq作为batch轴从而转换为batchMatmul。|默认非确定性实现，不支持配置开启| - |

## 废弃接口

|    废弃接口   |   说明     |
| --------------- | ----------------------- |
|[aclnnFusedInferAttentionScore](aclnnFusedInferAttentionScore.md)|此接口后续版本会废弃，请使用最新接口[aclnnFusedInferAttentionScoreV5](aclnnFusedInferAttentionScoreV5.md)。 |
|[aclnnFusedInferAttentionScoreV2](aclnnFusedInferAttentionScoreV2.md)|此接口后续版本会废弃，请使用最新接口[aclnnFusedInferAttentionScoreV5](aclnnFusedInferAttentionScoreV5.md)。 |
|[aclnnFusedInferAttentionScoreV3](aclnnFusedInferAttentionScoreV3.md)|此接口后续版本会废弃，请使用最新接口[aclnnFusedInferAttentionScoreV5](aclnnFusedInferAttentionScoreV5.md)。 |
|[aclnnGroupedMatMulAllReduce](aclnnGroupedMatMulAllReduce.md)|此接口后续版本会废弃，请勿使用该接口。|
|[aclnnIncreFlashAttention](aclnnIncreFlashAttention.md)|此接口后续版本会废弃，请使用最新接口[aclnnIncreFlashAttentionV4](aclnnIncreFlashAttentionV4.md)。 |
|[aclnnIncreFlashAttentionV2](aclnnIncreFlashAttentionV2.md)|此接口后续版本会废弃，请使用最新接口[aclnnIncreFlashAttentionV4](aclnnIncreFlashAttentionV4.md)。 |
|[aclnnIncreFlashAttentionV3](aclnnIncreFlashAttentionV3.md)|此接口后续版本会废弃，请使用最新接口[aclnnIncreFlashAttentionV4](aclnnIncreFlashAttentionV4.md)。 |
|[aclnnPromptFlashAttention](aclnnPromptFlashAttention.md)|此接口后续版本会废弃，请使用最新接口[aclnnPromptFlashAttentionV3](aclnnPromptFlashAttentionV3.md)。 |
|[aclnnPromptFlashAttentionV2](aclnnPromptFlashAttentionV2.md)|此接口后续版本会废弃，请使用最新接口[aclnnPromptFlashAttentionV3](aclnnPromptFlashAttentionV3.md)。 |
|[aclnnGroupedMatmul](aclnnGroupedMatmul.md)|此接口后续版本会废弃，请使用最新接口[aclnnGroupedMatmulV5](aclnnGroupedMatmulV5.md)。 |
|[aclnnGroupedMatmulV2](aclnnGroupedMatmulV2.md)|此接口后续版本会废弃，请使用最新接口[aclnnGroupedMatmulV5](aclnnGroupedMatmulV5.md)。 |
|[aclnnGroupedMatmulV3](aclnnGroupedMatmulV3.md)|此接口后续版本会废弃，请使用最新接口[aclnnGroupedMatmulV5](aclnnGroupedMatmulV5.md)。 |
|[aclnnGroupedMatmulV4](aclnnGroupedMatmulV4.md)|此接口后续版本会废弃，请使用最新接口[aclnnGroupedMatmulV5](aclnnGroupedMatmulV5.md)。 |
|[aclnnMlaProlog](aclnnMlaProlog.md)|此接口后续版本会废弃，请使用最新接口[aclnnMlaPrologV3WeightNz](aclnnMlaPrologV3WeightNz.md)。|
|[aclnnMlaPrologV2WeightNz](aclnnMlaPrologV2WeightNz.md)|此接口后续版本会废弃，请使用最新接口[aclnnMlaPrologV3WeightNz](aclnnMlaPrologV3WeightNz.md)。 |
|[aclnnMatmulAllReduceAddRmsNorm](aclnnMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnMatmulAllReduce](aclnnMatmulAllReduce.md)和[aclnnAddRmsNorm](../ops-nn/aclnnAddRmsNorm.md)。|
|[aclnnQuantMatmulAllReduceAddRmsNorm](aclnnQuantMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnQuantMatmulAllReduceV2](aclnnQuantMatmulAllReduceV2.md)和[aclnnAddRmsNorm](../ops-nn/aclnnAddRmsNorm.md)。|
|[aclnnWeightQuantMatmulAllReduceAddRmsNorm](aclnnWeightQuantMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnWeightQuantMatmulAllReduce](aclnnWeightQuantMatmulAllReduce.md)和[aclnnAddRmsNorm](../ops-nn/aclnnAddRmsNorm.md)。|
|[aclnnInplaceMatmulAllReduceAddRmsNorm](aclnnInplaceMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnMatmulAllReduce](aclnnMatmulAllReduce.md)和[aclnnAddRmsNorm](../ops-nn/aclnnAddRmsNorm.md)。|
|[aclnnInplaceQuantMatmulAllReduceAddRmsNorm](aclnnInplaceQuantMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnQuantMatmulAllReduceV2](aclnnQuantMatmulAllReduceV2.md)和[aclnnAddRmsNorm](../ops-nn/aclnnAddRmsNorm.md)。|
|[aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm](aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnWeightQuantMatmulAllReduce](aclnnWeightQuantMatmulAllReduce.md)和[aclnnAddRmsNorm](../ops-nn/aclnnAddRmsNorm.md)。 |
