MEMORY_TAGS: tuple[str, ...] = (
    # ------------------------------------------------------------
    # 1. Clinical / medical topics
    # ------------------------------------------------------------
    "glucose",              # 血糖总体：血糖值、趋势、空腹/餐后血糖
    "hypoglycemia",         # 低血糖风险或疑似低血糖：心慌、出汗、手抖、没吃饭后不适
    "hyperglycemia",        # 高血糖或偏高趋势：晨起偏高、餐后升高、控制不稳
    "medication",           # 用药本身：药名、用法、饭前饭后、胰岛素、二甲双胍、他汀等
    "medication_safety",    # 用药安全：漏服、重复服药、药物相互作用、不良反应、私自减药
    "medical_visit",        # 就医/复诊/检查/问诊/医生叮嘱
    "test_report",          # 报告单、检查指标、趋势图、血糖表格、化验结果解释
    "symptom",              # 一般症状：头晕、乏力、心慌、出汗、胸闷、腰酸等
    "neuropathy_foot",      # 糖尿病足/脚麻/腿麻/足部自查/破皮/发紫/冰凉
    "vision_eye",           # 眼部健康问题：视物模糊、眼底检查、糖尿病眼病风险
    "kidney",               # 肾功能、尿微量白蛋白、护肾、糖尿病肾病相关
    "complication",         # 糖尿病并发症泛化标签：神经、眼、肾、血管等并发症
    "safety_risk",          # 安全风险信号：跌倒、急性不适、伤口异常、严重症状
    "emergency_response",   # 紧急应对：快速联系家人、急诊、低血糖处理步骤、求助流程

    # ------------------------------------------------------------
    # 2. Self-management behaviors
    # ------------------------------------------------------------
    "diet",                 # 饮食、主食、碳水、加餐、家常饭、控糖饮食
    "activity",             # 运动、散步、八段锦、拍拍操、饭后活动
    "mobility",             # 行动能力、膝盖、走路、跌倒风险、手脚活动能力
    "sleep",                # 睡眠、午休、夜间、早醒、睡不好
    "monitoring",           # 监测行为：测血糖、测血压、足部检查、体重等
    "adherence",            # 依从性：忘测、忘吃药、坚持不了、执行困难
    "reminder",             # 提醒机制：闹钟、音箱、挂历、药盒提醒、家庭提醒
    "recording",            # 记录方式：本子、微信记录、语音记录、照片、录音、健康档案
    "sharing",              # 信息共享：给女儿看、微信群、给医生看、健康周报/简报
    "supplies",             # 耗材/药品补充：试纸、药快没了、买药、药盒、血糖仪
    "healthcare_access",    # 医疗服务流程：预约、挂号、医保、报销、异地备案、家庭医生
    "health_education",     # 健康知识学习：语音小课、广播式讲解、扫码看知识、科普内容

    # ------------------------------------------------------------
    # 3. Accessibility / interaction constraints
    # ------------------------------------------------------------
    "digital_literacy",     # 数字能力限制：不会复杂 App、怕按错、日历/待办/二维码困难
    "accessibility_vision", # 交互层面的视力限制：看不清小字、照片数字密、屏幕费眼
    "voice_interaction",    # 语音交互：像微信语音一样说、语音输入、语音播报、听而不是看
    "low_effort_workflow",  # 低负担流程偏好：少步骤、不确认、不填表、不额外记事
    "communication_style",  # 表达风格偏好：大白话、比喻、具体份量、少术语、少说教

    # ------------------------------------------------------------
    # 4. Family / social / emotional / lifestyle context
    # ------------------------------------------------------------
    "family",               # 女儿、儿子、老伴、孙辈、家庭微信群
    "caregiver",            # 照护者：女儿陪诊、老伴提醒、护工、邻居帮忙、社区人员
    "living_alone",         # 独居、一个人在家、无人即时帮忙
    "social",               # 社交、社区小组、群聊、熟人/陌生人互动
    "emotion",              # 情绪：心里闷、担心、不踏实、舒坦、孤单、需要倾听
    "care_burden",          # 怕麻烦家人、怕打扰孩子、照护压力、沟通负担
    "hobby",                # 兴趣爱好：听戏、评弹、纸鹤、针线活、手工、下棋、念经
    "routine",              # 日常规律：吃饭时间、听戏时段、午后习惯、复查周期
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
    "clinical_background",
    "stable_life_background",
    "functional_limitation",
    "recurring_health_pattern",
    "recurring_self_management_pattern",
    "care_context",
    "communication_accessibility_need",
    "solution_workflow_preference",
    "lifestyle_preference",
    "psychosocial_context",
)

BASIC_INFO_CLAIM_TYPE_DESCRIPTIONS = {
    "clinical_background": (
        "长期医学背景、长期用药、医生确认的慢病管理事实。不要记录单次不适。"
    ),

    "stable_life_background": (
        "相对稳定的生活、家庭、居住、日常背景，如与老伴同住、主要在社区医院复查。"
    ),

    "functional_limitation": (
        "长期或反复影响执行能力的身体、认知、操作限制，如看不清、记性差、膝盖差、不会打字。"
    ),

    "recurring_health_pattern": (
        "反复出现的身体/情绪状态模式，如多次脚麻、常担心血糖高。单次症状不要升级。"
    ),

    "recurring_self_management_pattern": (
        "反复出现的自我管理行为模式或困难，如常忘测血糖、常漏服药、不会坚持记录。"
    ),

    "care_context": (
        "照护关系和协作背景，如女儿陪诊、老伴提醒、邻居买药。"
    ),

    "communication_accessibility_need": (
        "信息表达方式需求，如大白话、大字、语音、少术语、具体份量。"
    ),

    "solution_workflow_preference": (
        "方案流程偏好，如不喜欢复杂 App、喜欢微信语音、不想点确认、能接受周报。"
    ),

    "lifestyle_preference": (
        "长期兴趣和生活偏好，如听戏、叠纸鹤、喜欢红薯。"
    ),

    "psychosocial_context": (
        "长期心理社会处境，如怕麻烦女儿、独居孤单、反复担心并发症。"
    ),
}


BASIC_INFO_STATUS: tuple[str, ...] = (
    "active",
    "uncertain",
    "superseded",
)