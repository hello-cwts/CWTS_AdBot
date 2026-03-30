# =========================
# config.py
# 所有可配置项集中在这里
# 非技术人员只需修改此文件
# =========================

# ---- 网站抓取 ----
SEED_URLS = [
    # 学院文化/历史/资格和人员介绍
    "https://www.cwts.edu/zh/about/statement-of-faith/",
    "https://www.cwts.edu/zh/about/feature/",
    "https://www.cwts.edu/zh/about/board/",
    "https://www.cwts.edu/zh/about/vision-and-mission-copy/",
    "https://www.cwts.edu/zh/about/future/",
    "https://www.cwts.edu/zh/about/administration/",
    "https://www.cwts.edu/zh/about/history/",
    "https://www.cwts.edu/zh/about/accreditation/",
    "https://www.cwts.edu/zh/about/ctlc/",
    "https://www.cwts.edu/zh/academic/faculty/",
    "https://www.cwts.edu/zh/academic/library/",
    "https://www.cwts.edu/zh/academic/catalog/",

    # 学位项目页
    "https://www.cwts.edu/zh/academic/degrees-programs/",
    # - 博士
    "https://www.cwts.edu/zh/academic/degrees-programs/doctor-of-ministry/",
    # - 硕士
    "https://www.cwts.edu/zh/academic/degrees-programs/master-of-theology/",
    "https://www.cwts.edu/zh/academic/degrees-programs/master-of-divinity/",
    "https://www.cwts.edu/zh/academic/degrees-programs/master-of-biblical-studies/",
    "https://www.cwts.edu/zh/academic/degrees-programs/master-of-christian-ministry/",
    "https://www.cwts.edu/zh/academic/degrees-programs/master-of-cross-cultural-leadership/",
    "https://www.cwts.edu/zh/academic/degrees-programs/master-of-theological-essentials/",
    # - 文凭
    "https://www.cwts.edu/zh/academic/degrees-programs/diploma-of-christian-studies/",
    "https://www.cwts.edu/zh/academic/degrees-programs/diploma-in-church-leadership/",
    # - 证书/事工学院
    "https://www.cwts.edu/zh/ministry-institute/catalog/",
    "https://www.cwts.edu/zh/ministry-institute/about/",
    "https://www.cwts.edu/zh/ministry-institute/application/",
    "https://www.cwts.edu/zh/ministry-institute/courses/",
    "https://www.cwts.edu/zh/ministry-institute/resources/",
    # - 线上项目
    "https://www.cwts.edu/zh/academic/online-program/",

    # 日程和教务章程
    "https://www.cwts.edu/zh/academic/course-schedule/",
    "https://www.cwts.edu/zh/academic/regulations/",
    "https://www.cwts.edu/zh/student-life/calendar/",

    # 申请资格
    "https://www.cwts.edu/zh/admissions/application-procedure/doctoral-degree/",
    "https://www.cwts.edu/zh/admissions/application-procedure/thm-degree/",
    "https://www.cwts.edu/zh/admissions/application-procedure/mccl/",
    "https://www.cwts.edu/zh/admissions/application-procedure/master/",

    # 国际学生
    "https://www.cwts.edu/zh/admissions/international/",

    # 奖学金和助学金
    "https://www.cwts.edu/zh/admissions/tuition-scholarship/",

    # 宿舍
    "https://www.cwts.edu/zh/student-life/dormitory/",
]

# 只跟踪 /zh/ 下包含这些关键词的内部链接
ALLOWED_PATH_KEYWORDS = [
    "/zh/admissions",
    "/zh/academic",
    "/zh/about",
    "/zh/ministry-institute",
    "/zh/student-life",
]

MAX_PAGES = 120         # 最多抓取页面数
REQUEST_DELAY = 0.5     # 每次请求间隔（秒）

# ---- 向量库 ----
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 800        # 每段字数
CHUNK_OVERLAP = 100     # 段落重叠字数
EMBEDDING_MODEL = "text-embedding-3-small"

# ---- GPT 模型 ----
GPT_MODEL = "gpt-4o-mini"
GPT_TEMPERATURE = 0.3

# ---- Google Sheet ----
# Sheet A：用户注册记录
SIGNUP_SHEET_TAB = "signups"

# Sheet B：Q&A知识库 + 未解问题（同一个文件，两个Tab）
QA_SHEET_TAB = "qa_bank"            # Tab1：staff维护的Q&A库
PENDING_SHEET_TAB = "pending"       # Tab2：待审列表（GPT回答过的）
UNANSWERED_SHEET_TAB = "unanswered" # Tab3：完全未解问题