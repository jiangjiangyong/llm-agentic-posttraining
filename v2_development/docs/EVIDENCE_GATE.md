# Evidence Gate 模板

每个新阶段建立一个独立的 gate 记录，至少回答：

1. 解决了什么明确问题？
2. Baseline 是什么？
3. 指标是什么？
4. 有没有量化收益？
5. 增加了多少 step、token、latency 或存储成本？
6. Failure 分布如何变化？
7. 用什么命令和版本可以复现？

## 当前 Gate：CPU Runtime Foundation

- 问题：在没有 GPU/真实模型时，先验证 Runtime 的状态、工具、checkpoint 和 recovery 契约。
- Baseline：确定性的 `MockModelAdapter` + calculator/workspace tools。
- 指标：run status、tool calls、state transition count、checkpoint count、failure/recovery count、steps。
- 结论边界：仅证明 CPU 工程链路可运行；不证明模型能力、训练收益或真实任务成功率。
- 复现：`python -m unittest discover -s . -p "test_*.py" -v`；`python scripts/smoke_cpu.py`。
