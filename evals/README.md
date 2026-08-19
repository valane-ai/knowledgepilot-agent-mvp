# 检索评测

复制 `qa_dataset.example.jsonl` 为 `qa_dataset.jsonl`，每一行填写一个真实任务中的问题、正确资料文件名和期望关键词。运行：

```powershell
python evals/run_retrieval_eval.py --dataset evals/qa_dataset.jsonl --k 5
```

脚本输出 Recall@K 与 MRR。答案准确率和 Faithfulness 需要在同一数据集补充 `reference_answer` 后，以人工抽检或独立 LLM 裁判评测；不要让被测模型同时充当唯一裁判。
