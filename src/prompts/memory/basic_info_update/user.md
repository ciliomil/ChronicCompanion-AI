以下是上一版 basic_info、本 session 新抽取的事件列表和需求-方案条目。

请根据系统提示更新长期背景画像 basic_info，并只返回 JSON 对象。

上一版 basic_info：
$old_basic_info

本 session 新抽取的 EventItem：
$session_events

本 session 新抽取的 NeedItem：
$session_need_items

注意：
- EventItem 是主要事实证据来源。
- NeedItem 可提供需求背景、长期约束和稳定生活偏好线索，但不要把 AI 方案本身写入 basic_info。
- 如果没有足够长期性证据，请不要新增 claim。
