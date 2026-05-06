以下是本 session 新抽取的 EventItem 与 NeedItem。

请按系统提示提出长期背景候选 claim，并只返回 JSON 对象。

本 session 新抽取的 EventItem（每行末尾的 [tags: ...] 即该 event 可选的 tag 范围）：
$session_events

本 session 新抽取的 NeedItem：
$session_need_items

注意：
- 候选 claim 的 tags 必须从其引用的 EventItem 的 tags 和 NeedItem 的 related_tags 的并集里保守选择 1~3 个，不要使用其他 tag。
- 不要参考任何旧 basic_info 内容；本阶段只负责"提出候选"。
- 若没有任何足够长期性的证据，请返回 {"candidates": []}。
