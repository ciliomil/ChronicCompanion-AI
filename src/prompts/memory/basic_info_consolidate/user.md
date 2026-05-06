以下是 basic_info 已有的 active 和 uncertain claims（按 section 分组）以及本 session 新提出的候选 claims。

请按系统提示对**每一个 candidate**给出一个 action，并只返回 JSON 对象。

已有 basic_info claims（仅列 active 和 uncertain）：
$existing_claims

本 session 候选 claims：
$candidates

注意：
- target_claim_id 仅可使用上方"已有 basic_info claims"中真实出现的 claim_id。
- merged_tags（如果给出）必须从目标 claim tags 与 candidate tags 的并集中保守选择 1~3 个。
- 已有 claim 不在 decisions 中提及时，系统会保持其不变。
