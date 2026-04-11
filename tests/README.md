# 测试 README

根据提供的测试示例，以下是用于记录和测试 MMR（最大边际相关性）选择功能的单元测试：

## 测试组织

测试按功能进行组织：
- **基本功能**：验证核心 MMR 行为
- **边界情况**：处理边界条件
- **参数验证**：测试不同的 lambda 值
- **多样性验证**：确保相关性-多样性权衡

## 测试覆盖

1. **test_mmr_returns_k** - 验证 MMR 返回恰好 k 个结果
2. **test_mmr_fewer_than_k** - 处理少于 k 个文档的情况
3. **test_mmr_diversity** - 验证多样性选择机制
4. **test_mmr_lambda_1_equals_greedy** - 确认 lambda=1.0 表现为贪心选择
5. **test_mmr_lambda_0_equals_diversity** - 测试 lambda=0.0 强调多样性
6. **test_mmr_empty_docs** - 处理空文档列表
7. **test_mmr_single_doc** - 当 k=1 时返回单个文档
8. **test_mmr_invalid_k** - 验证 k 参数约束
9. **test_mmr_vector_dimension_mismatch** - 处理维度不匹配
10. **test_mmr_score_ordering** - 验证基于分数的初始排序

## 说明

- 所有测试使用模拟的 asyncpg 连接来避免数据库依赖
- 测试关注功能正确性，而不是性能
- 向量操作使用 numpy 以确保数值稳定性
- 异步测试使用 `@pytest.mark.asyncio` 标记