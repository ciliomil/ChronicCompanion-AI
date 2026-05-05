MEMORY_TAGS: tuple[str, ...] = (
    "glucose",
    "medication",
    "medical_visit",
    "symptom",
    "complication",
    "diet",
    "sleep",
    "activity",
    "monitoring",
    "adherence",
    "family",
    "caregiver",
    "living_alone",
    "social",
    "emotion",
    "stress",
    "hobby",
    "safety_risk",
    "other",
)

NEED_DOMAINS: dict[str, str] = {
    "diet_glucose_management": "饮食、加餐、主食、饮食对血糖影响",
    "glucose_monitoring_recording": "血糖测量、记录、趋势理解、监测解释",
    "medication_adherence": "用药提醒、服药确认、漏服、用药记录",
    "symptom_risk_triage": "症状担忧、并发症、副作用、何时就医",
    "activity_safety": "运动、居家活动、膝盖友好、安全活动",
    "routine_habit_adherence": "健康习惯提醒、生活节奏绑定、坚持习惯",
    "family_caregiver_collaboration": "家属/照护者知情、协作、共享、减轻压力",
    "healthcare_navigation": "复诊准备、医嘱留存、医保/补贴/社区服务流程",
    "supplies_device_management": "试纸、助听器、电池、设备耗材、补货预防",
    "emotional_motivation": "情绪支持、鼓励、认可、趣味激励、解闷",
    "daily_life_task_support": "邻里帮助、生活事务记录、工具操作交接",
    "other": "其他",
}

BASIC_INFO_CATEGORIES: tuple[str, ...] = (
    "medical_care",
    "family",
    "health",
    "leisure",
)

BASIC_INFO_CLAIM_TYPES: tuple[str, ...] = (
    "stable_fact",          # 明确长期事实：用户有糖尿病、与老伴同住
    "recurring_pattern",    # 反复出现的模式：常忘记测糖、经常饭后散步
    "long_term_constraint", # 长期约束：膝盖不好、不便出远门
    "long_term_preference", # 现实生活偏好：不爱吃甜食、喜欢下棋
    "care_context",         # 照护背景：女儿常提醒饮食、老伴负责陪诊
)

BASIC_INFO_STATUS: tuple[str, ...] = (
    "active",
    "uncertain",
    "superseded",
)