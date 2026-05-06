以下是 basic_info 四个 section 合并后的 active 和 uncertain claims（按 section 分组）。

请按系统提示为每个 section 生成 1~3 句中文 summary，并只返回 JSON 对象。

$section_claims

注意：
- summary 只能依据上面列出的 claims，不要扩展未出现的事实。
- 没有 claim 的 section，summary 返回空字符串，summary_source_claim_ids 返回空数组。
