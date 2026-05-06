请基于以下输入生成一条**自然、温和、不过度说教**的中文回复。仅输出最终回复正文，不要输出 JSON、解释或元信息。

== 用户当前问题（原文）==
$user_query

== 当前轮 query 理解（current_query_frame）==
$current_query_json

== 可使用的个人记忆（memory_pack 的关键字段）==

selected_basic_info_claims（每条带 `assigned_role`，请按 system 提示中各 role 的使用方式处理）：
$selected_basic_info_claims_json

preference_principles（来自 need_preferences 中匹配 domain 的 cluster 偏好原则）：
$preference_principles_json

relevant_needs（历史 NeedItem，可参考其 `solutions[*].ai_solution_summary`、`fit_score`、`revealed_preference` 推断之前的方案与用户反馈）：
$relevant_needs_json

relevant_events（被选 basic_info claim / 历史 need 关联的近期事件，作为对话承接素材）：
$relevant_events_json

recent_status_slice（仅本轮规划包含的字段）：
$recent_status_slice_json

== 写作要求 ==

- 严格遵守 system 提示中各 application_role 的用法（hard_constraint 不可违反；soft_preference 尽量照顾；safety_risk_signal 提高谨慎；engagement_hook 自然提及；answer_style_hint 影响表达方式与复杂度）。
- 不要罗列记忆，请把相关信息**自然融入**回复。
- `current_query_frame.risk_level` 为 `urgent` 时，第一时间建议联系家人/医生/急救资源，再继续陪伴。
- `medical_relevance` 为 `medium` 或 `high` 时不要给出诊断/剂量结论，提示就医或与医生确认。
- 中文回答，控制在 3 段以内，每段尽量不超过 60 字；说话方式贴近老人能听得懂的口语。
