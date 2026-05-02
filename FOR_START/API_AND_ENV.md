# API_AND_ENV.md

## 使用说明
本文件用于告诉 AI 编码助手：
- 当前项目可能会用到哪些接口
- 需要预留哪些适配层
- 环境变量如何组织
- 哪些技术选择需要保持克制

> 注意：如果仓库里尚未最终确定某个接口，请先按“适配层 + 占位实现”的方式设计，不要把业务逻辑直接写死在某家服务上。

---

## 1. 推荐总体原则
- LLM 调用统一封装在 `llm_client` 或等价模块中
- Embedding 调用统一封装在 `embedding_client` 或等价模块中
- 向量检索封装在 `vector_store` 抽象层中
- 配置统一从环境变量读取
- 不要把 key、base_url、model_name 写死在业务代码中

---

## 2. LLM 接口
当前项目需要至少支持一个大模型调用接口，用于：
- 回答生成
- 事件抽取
- 需求-方案抽取
- 需求推测
- 画像更新（后续）
- 安全辅助判断（可选）

### 建议适配方式
统一使用类似如下抽象：
- `generate_text(prompt, system_prompt=None, temperature=..., ...)`
- `generate_json(prompt, schema_hint=..., ...)`


### 参考
- **请你参考wxf/longmem_test/naive_mem0/conf.yaml完成api配置**
---

## 3. Embedding 接口
项目中的 baseline 或后续检索实验可能需要 embedding，用于：
- 原始历史 chunk 向量化
- baseline 向量检索
- 相关性召回辅助

### 参考
- **请你参考wxf/longmem_test/naive_mem0/conf.yaml完成api配置**

---

## 4. 向量存储
如果实现通用向量检索式 baseline，建议通过抽象层支持向量库。

可先支持：
- 内存 mock / 本地轻量实现
- 简单文件存储
- 预留 Qdrant / Chroma / FAISS 等适配空间

### 参考
- **请你参考wxf/longmem_test/naive_mem0的向量存储完成配置**

---

## 5. 结构化存储
三层长期记忆中，除向量检索外，还需要结构化存储。

建议：
- 原型阶段优先使用本地轻量存储
- 可选 JSON / SQLite
- 重点是结构清晰、易调试、易实验，而不是一开始上复杂数据库

### 建议环境变量
- `STORAGE_TYPE`
- `SQLITE_PATH`
- `DATA_DIR`

---

## 6. 应用服务
建议后端采用轻量服务方式，优先支持：
- 一个基础聊天接口
- 一个健康检查接口
- 可选的调试接口

### 建议环境变量
- `APP_ENV`
- `APP_HOST`
- `APP_PORT`
- `LOG_LEVEL`

---

## 7. 实验开关
项目需要方便切换不同实验模式。

建议环境变量或配置项包括：
- `ENABLE_SAFETY`
- `ENABLE_NEED_INFERENCE`

---

## 8. 目录和配置建议
建议将配置集中在：
- `.env`
- `config.py` 或 `settings.py`

建议通过统一配置对象暴露：
- 模型名称
- provider
- base_url
- api_key
- 存储路径
- 实验模式
- 功能开关

---

## 9. AI 编码助手注意事项
- 若仓库里尚未明确具体 provider，先写抽象接口，不要绑定某一家平台
- 若某项外部服务暂未配置，先提供 stub / mock 实现
- 配置读取与业务逻辑要分离
- 所有第三方依赖都应尽量最小化
- 先保证本地可运行，再考虑可扩展性

---

## 10. 推荐的最小实现策略
在未确认完整接口前，建议先这样做：
- 用一个统一的 `LLMClient` 包装文本生成
- 用一个 `MockEmbeddingClient` 或可替换 embedding 接口
- 用本地 JSON / SQLite 先实现三层记忆存储
- 用内存或本地简单索引先跑通 baseline
- 后续再替换真实向量库或更强模型

这样更适合当前科研原型阶段。
