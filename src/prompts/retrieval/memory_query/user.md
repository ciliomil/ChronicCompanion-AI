请只输出一个 JSON 对象，形状与 system 提示一致（仅含 MemoryQuery 相关字段，不要输出 safety_notes）。

当前用户问题
$user_query

对话上下文（可为空；可为多行文本）
$dialogue_context

## 用户 basic_info — 四类 summary（JSON）

$basic_info_summaries_json

## 用户 basic_info — 各类 claims（仅列出 claim_id、content、claim_type、source_event_ids；用于你选择 basic_info_claim_ids）
$basic_info_claims_json

用户 recent_status
$recent_status
