# BUILD_TASKS.md

## 当前目标：先实现 layered 与 baseline 的可运行骨架

### 任务 1：建立适合实验的项目目录结构
目标：
- 让后续对比实验、消融实验更容易开展

建议结果：
- `app/` 或 `src/` 目录
- `memory/`
- `retrieval/`
- `dataset/`
- `experiments/`
- `llm/`
- `storage/`
- `api/` 或 `runner/`

---

### 任务 2：建立统一配置系统
目标：
- 支持通过配置切换运行模式
- 支持后续切换模型、数据路径、存储方式、是否启用安全模块等

至少支持以下配置项：
- `DATASET_PATH`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `ENABLE_SAFETY`
- `ENABLE_NEED_INFERENCE`
- `OUTPUT_DIR`

产出：
- `.env.example`
- `config.py` 或 `settings.py`

---

### 任务 3：实现 baseline 的可运行代码

当前要在experiment中至少实现两个baseline：
1. **no_memory**
   - 仅基于当前输入生成回答
2. **vector_memory_baseline**
   - 基于历史 chunk 做简单检索再生成回答
   - **参考wxf/longmem_test/naive_mem0**

数据集格式和测评任务、测评方法参考wxf/Mem-PAL，对于当前老年慢病患者场景我参考Mem-PAL造的具体数据存在wxf/accompany_dataset中，还未完成，但格式相同。

---

