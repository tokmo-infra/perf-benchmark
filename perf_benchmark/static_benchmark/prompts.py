"""Pre-crafted long-generation coding prompts designed for sustained token generation.

These prompts require in-depth thinking, architectural planning, and zero-omission,
production-grade code implementation across diverse domains (games, voxel engines,
distributed systems, OS kernels, databases, rendering engines) to ensure the model
generates up to the maximum completion token limit (~2560 tokens) without premature EOS.
"""

from __future__ import annotations

from typing import Sequence

# Long-generation system instruction demanding deep reasoning and complete code
SYSTEM_PROMPT_LONG_CODE = (
    "你是一位资深架构师与底层系统核心工程师。对于用户的编码需求，请必须遵循以下两阶段流程：\n"
    "【第一阶段：思考与架构规划】深入推演并详细阐述架构设计、模块拆分、核心状态机、关键数据结构及算法复杂度，覆盖边界条件与容错处理。\n"
    "【第二阶段：零省略完整源码实现】架构思考完成后，直接给出完整连贯、工业级标准的代码。绝对严禁任何形式的省略（如 // TODO、// 篇幅原因省略、// ... 其余逻辑类似 等），必须写出所有辅助函数、错误处理及可运行的完整逻辑。"
)

CODING_PROMPTS: list[dict[str, str]] = [
    {
        "id": "voxel_minecraft_rust",
        "title": "Rust 终端字符 3D 体素沙盒引擎《迷你我的世界》(Voxel Engine)",
        "prompt": (
            "请用 Rust 语言从零实现一个可以在终端字符界面运行的轻量级 3D 体素引擎与沙盒游戏《我的世界》。\n"
            "具体实现要求如下：\n"
            "1. 架构推导与规划：详细推导体素块类型定义（空气、泥土、草方块、石头、木头、水、黑曜石等属性）、"
            "Chunk 数据结构（16x16x16 体素切片）、基于 3D Simplex/Perlin 噪声的高度图地形与洞穴生成算法。\n"
            "2. 射线步进与视线求交：实现快速体素遍历算法（Fast Voxel Traversal / 3D DDA 射线步进），"
            "精确计算玩家准星所对准的方块坐标以及相邻放置面法线。\n"
            "3. 玩家物理引擎：包含玩家位置、视角（俯仰角 Pitch、偏航角 Yaw）、重力加速度、跳跃推力、"
            "AABB 包围盒与周围体素碰撞检测及滑动摩擦力反应。\n"
            "4. 终端字符渲染管线：建立三维透视投影变换，利用射线投射（Ray Casting）或多边形光栅化将 3D 场景"
            "光栅化为终端字符帧缓冲区（FrameBuffer），结合光源衰减与距离深度雾化（使用不同灰度字符如 '@%#*+=-:. '）。\n"
            "5. 交互与游戏循环：支持 WASD 移动、空格跳跃、鼠标或按键旋转视角、左键破坏方块、右键放置方块，以及游戏循环的动态帧率控制。\n"
            "请先进行严密的架构推导与规划，随后直接提供完整、无任何省略的全部 Rust 代码。"
        ),
    },
    {
        "id": "snake_ai_cpp20",
        "title": "C++20 高性能跨平台《贪吃蛇》与双模式 A*/Hamilton 寻路 AI 引擎",
        "prompt": (
            "请使用现代化 C++20 编写一个功能完备的高性能跨平台终端《贪吃蛇》(Snake) 游戏系统与双模式自动寻路 AI 对战引擎。\n"
            "具体实现要求如下：\n"
            "1. 架构思考与规划：深入分析游戏主循环（Game Loop）、双缓冲无闪烁 ANSI 终端转义序列渲染器、"
            "高精度纳秒级计时器与固定步长物理更新逻辑。\n"
            "2. 贪吃蛇核心逻辑：采用环形缓冲区或双向链表高效维护蛇身节点，实现蛇头前进、转向缓冲队列、"
            "不同机制食物（普通食物、加分金色苹果、减速冰冻果、缩短蛇身毒菇）随机分布与有效碰撞判定。\n"
            "3. 自动寻路 AI 算法：实现两套完备的 AI 算法：\n"
            "   a. 基于 A* 启发式搜索的最短安全进食路径算法，具备防死锁曼哈顿距离预估与未来身体障碍物动态投影；\n"
            "   b. 基于哈密顿回路（Hamiltonian Cycle）的全图安全漫游保底算法，在空间狭窄时自动切换，确保理论上可吃满全图且永不自撞。\n"
            "4. 多蛇对战与动态关卡：支持人机对战模式，包含蛇与蛇之间的头身相撞判定、击杀奖励、动态随机障碍物障碍墙生成。\n"
            "5. 统计与回放系统：完整的得分统计、历史操作回放压缩编码与解析输出。\n"
            "请先进行深度的算法设计推演，随后给出完全连贯、绝对无任何省略的完整 C++ 源代码。"
        ),
    },
    {
        "id": "tetris_srs_typescript",
        "title": "TypeScript 工业级 Tetris Guidelines 规范《俄罗斯方块》与自动化求解器",
        "prompt": (
            "请使用 TypeScript 从零编写一个完全符合现代 Tetris Guidelines 官方规范的《俄罗斯方块》(Tetris) 核心引擎与启发式自动求解器。\n"
            "具体实现要求如下：\n"
            "1. 规划与规格思考：Playfield 矩阵设计（10列 x 20行可见区域 + 20行缓冲区）、完整的 7-Bag 随机块生成器（确保每个循环包含七种经典方块 I, J, L, O, S, T, Z 且分布均匀）。\n"
            "2. 超级旋转系统（Super Rotation System, SRS）：完整实现 SRS 旋转规则，包括顺时针/逆时针旋转矩阵、"
            "完整的 Wall Kick 偏移判定表（详细列出 I 块 5 组状态转移偏移与 JLSTZ 块 5 组状态转移偏移的所有测试坐标）。\n"
            "3. 方块操作与物理：重力下落速度曲线、锁定延迟（Lock Delay，每次有效移动/旋转重置最多15次）、"
            "暂存功能（Hold）、硬降（Hard Drop 直接落地并锁定）、软降（Soft Drop）、Ghost 幽灵阴影投影计算。\n"
            "4. 判定与计分系统：行消除逻辑、T-Spin（3-Corner 顶点判定与 Mini T-Spin 识别）、Back-to-Back (B2B) 连续特殊消除加成、Combo 连击倍率计算。\n"
            "5. 启发式自动落块 AI：基于 Pierre Dellacherie 评估算法，实时评估井深（Wells）、空洞数（Holes）、高度差异（Bumpiness）和表面平整度，自动搜索当前最佳落点与旋转次数。\n"
            "请先进行严密的规范推导，然后输出零省略、可直接编译运行的全部 TypeScript 源码。"
        ),
    },
    {
        "id": "raft_consensus_go",
        "title": "Go 生产级轻量级分布式 Raft 共识协议状态机",
        "prompt": (
            "请用 Go 语言从零实现一个工业级、功能完备的轻量级分布式 Raft 共识算法引擎与强一致性 KV 存储状态机。\n"
            "具体实现要求如下：\n"
            "1. 架构思考与推演：深入阐述 Leader 选举、Term 任期机制、心跳保活、日志复制、安全性约束（选举安全、Leader 日志完整性、状态机安全定理）。\n"
            "2. 节点核心结构体：定义 RaftNode，包含 PersistentState（currentTerm, votedFor, log 条目）、"
            "VolatileState（commitIndex, lastApplied）、LeaderState（nextIndex, matchIndex 切片）、角色状态迁移枚举。\n"
            "3. 核心 RPC 交互逻辑：\n"
            "   a. RequestVote RPC 处理端与发起端：候选人自增任期向所有节点并发拉票，严格检查候选人日志的新旧程度（LastLogIndex/LastLogTerm）；\n"
            "   b. AppendEntries RPC 处理端与发起端：心跳广播、日志一致性检测（PrevLogIndex/PrevLogTerm 匹配）、冲突条目截断与新条目追加、更新 commitIndex。\n"
            "4. 日志压缩与快照（Log Compaction & Snapshot）：内存快照创建、日志截断逻辑，以及 InstallSnapshot RPC 跨节点快照同步。\n"
            "5. 强一致性 KV 存储状态机：客户端 Put/Get 请求处理通道、线性一致性读、状态机 apply 协程解耦与防死锁并发控制。\n"
            "请先详细推导选举与日志复制状态机，随后直接提供完整且无任何省略的全部 Go 代码。"
        ),
    },
    {
        "id": "rtos_scheduler_c",
        "title": "ANSI C 嵌入式实时操作系统 (RTOS) 抢占式微内核调度器",
        "prompt": (
            "请使用 ANSI C 语言编写一个高可靠嵌入式实时操作系统（RTOS）微内核的核心任务调度器与内存管理子系统。\n"
            "具体实现要求如下：\n"
            "1. 架构思考与推演：任务控制块（TCB）、任务生命周期状态（Ready, Running, Blocked, Suspended）、"
            "多优先级就绪队列（支持 32 个优先级，基于位图快速查找最高优先级的 O(1) 调度算法，结合 __builtin_clz 或软件查表法）。\n"
            "2. 调度与上下文切换机制：任务上下文保护与恢复框架（保存通用寄存器与程序计数器栈帧结构）、"
            "SysTick 时钟滴答中断处理、基于时间片的轮转调度（Round-Robin for same priority）以及休眠延迟链表（Delta Delay List）递减唤醒机制。\n"
            "3. 同步与互斥原语：\n"
            "   a. 互斥量（Mutex）与优先级继承协议（Priority Inheritance Protocol），彻底规避经典优先级反转问题；\n"
            "   b. 计数信号量（Semaphore）与事件标志组（Event Flags），支持任务挂起等待队列唤醒。\n"
            "4. 内存管理：固定块大小内存池（Memory Pool）与可变尺寸伙伴内存分配器（Buddy System），具备内存碎片统计与越界保护标志。\n"
            "5. 启动与验证：内核初始化函数、任务创建 API（TaskCreate）、调度器启动例程与两个并发演示任务。\n"
            "请先进行详尽的内核设计推导，随后提供完整连贯、零省略的全部 C 语言代码。"
        ),
    },
    {
        "id": "tsdb_storage_python",
        "title": "Python 高性能内存时序数据库 TSDB 核心与 Gorilla 压缩算法",
        "prompt": (
            "请用 Python (纯标准库与二进制 struct/bytearray) 实现一个高性能内存时序数据库（TSDB）核心存储引擎与聚合计算系统。\n"
            "具体实现要求如下：\n"
            "1. 压缩算法原理与推导：深入推导并完整实现 Facebook Gorilla 论文的核心压缩技术：\n"
            "   a. 时间戳 Delta-of-Delta 可变长比特流编码（针对固定间隔采集的极高压缩比）；\n"
            "   b. 浮点数值 XOR 差值压缩（利用 IEEE 754 双精度浮点格式，记录前导零与尾随零可变长编码比特位）。\n"
            "2. 存储引擎架构：\n"
            "   a. 序列倒排索引（Inverted Index）：支持按指标名（Metric Name）与标签键值对（Labels / Tags）快速匹配时间序列 ID；\n"
            "   b. 分块时间存储（Chunk Manager）：将时间划分为固定窗口（如 2 小时），活跃块（Active Chunk）持续写入，封存块（Closed Chunk）转为只读压缩二进制段。\n"
            "3. WAL 预写日志与恢复：所有写入点同步记录 WAL，支持数据库意外崩溃后的冷启动快速重放恢复。\n"
            "4. 范围查询与向量化聚合器：支持给定时间范围 [start, end] 快速提取解压时序数据，"
            "提供包含 Count, Sum, Min, Max, Avg, Rate (变化率), P50/P90/P99 百分位数的滑动窗口聚合算子。\n"
            "5. 并发安全与清理：读写锁保护（RWLock）、数据生命周期 TTL 自动淘汰机制。\n"
            "请先进行深度的比特流压缩算法与索引推导，随后给出完整无缺、工业级规范的全部 Python 源码。"
        ),
    },
    {
        "id": "path_tracer_cpp20",
        "title": "C++20 纯 CPU 蒙特卡洛路径追踪光线渲染器 (Path Tracer)",
        "prompt": (
            "请使用现代化 C++20 编写一个无任何第三方依赖的高性能多线程 CPU 蒙特卡洛路径追踪光线渲染器（Ray / Path Tracer）。\n"
            "具体实现要求如下：\n"
            "1. 物理光学推导：详细推导辐射度学基础、Kajiya 渲染方程、半球均匀采样与余弦加权半球重要性采样、BRDF 双向反射分布函数公式。\n"
            "2. 基础数学几何库：高效三维向量 Vector3（点乘、叉乘、反射、折射、规范化）、Ray 射线结构体、AABB 轴对齐包围盒。\n"
            "3. 空间加速层次结构（BVH）：基于表面积启发式（SAH - Surface Area Heuristic）构建高效的层次包围盒树，"
            "支持图元划分、递归构建与基于栈展开的高速光线求交遍历遍历（Ray-BVH Intersection）。\n"
            "4. 几何体与材质系统：\n"
            "   a. 球体与三角形网格求交（Möller-Trumbore 算法）；\n"
            "   b. 漫反射材质（Lambertian Diffuse）；\n"
            "   c. 金属高光反射材质（Metal with Roughness）；\n"
            "   d. 绝缘透明介质（Dielectric 玻璃，精确实现折射 Snell 定律与菲涅尔近似 Schlick 方程）；\n"
            "   e. 自发光漫射光源（Diffuse Light）。\n"
            "5. 相机与多线程渲染管线：具备景深（焦距与光圈）、视野（FOV）的可调相机，"
            "使用 std::jthread 与分块任务队列并行渲染，支持超采样抗锯齿（MSAA），最后直接将色调映射（Gamma 校正）后的图像保存为 PPM 格式。\n"
            "请先进行物理与几何算法推导，随后直接提供零省略、可直接编译运行的全部 C++ 源码。"
        ),
    },
    {
        "id": "relational_db_engine_python",
        "title": "Python 关系型数据库存储引擎 (B+ 树分页) 与 Volcano 迭代器查询执行器",
        "prompt": (
            "请用 Python 从零构建一个微型关系型数据库核心引擎，包含底层的 B+ 树物理分页存储与上层 Volcano（火山模型）迭代器查询执行引擎。\n"
            "具体实现要求如下：\n"
            "1. 架构规划与推演：梳理磁盘文件页抽象（Page 4KB 对齐）、LRU 缓冲池管理器（Buffer Pool Manager，维护 Frame、Dirty Page 刷盘机制与 Pin/Unpin 引用计数）。\n"
            "2. 完整物理 B+ 树索引实现：\n"
            "   a. 定义 BPlusTreeInternalPage 与 BPlusTreeLeafPage 节点结构；\n"
            "   b. 实现键值搜索、插入操作（包含当节点键数量达到上限阶数时的递归分裂 Split 逻辑）；\n"
            "   c. 实现删除操作（包含节点借键 Borrow 与节点合并 Merge 逻辑，保证树的平衡性）；\n"
            "   d. 叶子节点双向链表遍历以支持高速范围扫描。\n"
            "3. 表目录与元数据：TableSchema 列定义（整数、浮点、定长字符串）、Tuple 行记录序列化与二进制反序列化。\n"
            "4. 火山模型执行器算子：实现统一迭代器接口（open, next, close），包含：\n"
            "   a. SeqScan（全表顺序扫描）；\n"
            "   b. IndexScan（基于 B+ 树的快速点查与范围过滤）；\n"
            "   c. Filter（谓词表达式条件过滤）；\n"
            "   d. NestedLoopJoin 与 HashJoin（双表关联算子）；\n"
            "   e. Aggregate（支持 Group By 聚合，如 Count, Sum, Avg）。\n"
            "5. 简单查询编译运行示例：演示建表、插入测试数据、构建执行计划树并流式迭代输出查询结果。\n"
            "请先进行清晰的系统架构推导，随后输出完整、连贯、没有任何省略的工业级 Python 代码。"
        ),
    },
]


def get_decode_prompt(index: int, custom_prompts: Sequence[str] | None = None) -> tuple[str, str]:
    """Retrieve a long coding prompt for the given request index.

    Returns:
        tuple[str, str]: (prompt_text, system_instruction)
    """
    if custom_prompts and len(custom_prompts) > 0:
        idx = index % len(custom_prompts)
        return custom_prompts[idx], SYSTEM_PROMPT_LONG_CODE

    item = CODING_PROMPTS[index % len(CODING_PROMPTS)]
    return item["prompt"], SYSTEM_PROMPT_LONG_CODE


def get_all_prompts_info() -> list[dict[str, str]]:
    """Return catalog of available built-in long-coding prompts."""
    return [{"id": p["id"], "title": p["title"]} for p in CODING_PROMPTS]
