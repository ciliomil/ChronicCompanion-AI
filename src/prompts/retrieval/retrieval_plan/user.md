请只输出一个 JSON 对象，结构为：
{"current_query_frame": {...}, "memory_retrieval_plan": {...}}

当前用户问题：
$user_query

对话上下文（可为空；可多行）：
$dialogue_context

用户 basic_info（含 4 个 section 的 summary 与 active/uncertain claims）：
$basic_info_json

用户 recent_status：
$recent_status_json

注意：
- selected_basic_info_claims 中的 claim_id 仅能使用上方 basic_info 中真实存在的 claim_id。
- frame 与 plan 不要重复表达语义内容；plan 只放选定 id、预算、字段子集。
- 严格保守地选择 claim 与召回预算；本轮用不上的不要选。
