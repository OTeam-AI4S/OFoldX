# TikTok Lessons

## 用途

记录 TikTok 流程中经过 Tik 确认、可复用于后续任务的经验。普通执行流水、未经验证的推测和仅对单次任务有意义的细节不应写入。

## 记录规则

- 条目 ID：`LESSON-YYYYMMDD-NNN`
- Tik 提供或确认内容，Tok 写入，Tik 复审。
- 条目应包含证据、适用范围和可执行建议。
- 不静默删除或重写历史；纠正时追加更正说明。
- 写入前检查是否已有相同经验。

## Entries

### LESSON-20260728-001: 分侧固定解释器与导入根

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 背景：本地只有 `base` 与 `ofantibody` 两个 Conda 环境，且两个源码树不在脚本启动时的默认导入根中。
- 经验：upstream 与 OFoldX 必须在独立进程中固定解释器、`PYTHONPATH` 和源码 commit，不能用一个环境探测结果替代另一侧。
- 证据：`base` 缺少 numpy、torch、biotite、rdkit、tensordict、ml_collections；`ofantibody` 仅缺 tensordict。upstream 需要 `PYTHONPATH=baseline/Protenix`，OFoldX 需要仓库根目录。
- 适用范围：所有比较两个独立源码树或依赖环境的数据 parity harness。
- 后续行动：每侧单独运行 diagnose/export，并在 artifact 中记录 Python executable、`PYTHONPATH` 与 commit。
- Tik 复审：2026-07-28 Tik 最终复审确认。
- 更正记录：无。

### LESSON-20260728-002: 重复失败比较结构化指纹

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 背景：同一失败的原始 stderr 会受 Conda footer、repeat 参数和临时缓存提示影响。
- 经验：失败重复性应比较 exception type/message、side、sample 和环境信息，不比较日志字节。
- 证据：upstream `1zev` 两次原始日志因 Matplotlib 临时缓存提示不同，但对应 `failure.json` SHA256 一致。
- 适用范围：所有预期失败或环境阻塞场景的重复性验证。
- 后续行动：保留原始日志用于诊断，以结构化 `failure.json` 作为重复失败判据。
- Tik 复审：2026-07-28 Tik 最终复审确认。
- 更正记录：无。

### LESSON-20260728-003: 先核对声明 floor 与官方可用版本

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 背景：OFoldX 声明 `chanfig>=0.0.117`，但官方 PyPI、官方仓库 `master` 和官方 tag 的最高版本均为 `0.0.116`。
- 经验：建立 clean room 验证环境前，应先核对依赖声明的版本 floor 是否存在官方可获取构件；官方来源不满足声明时，严格通道必须阻塞，禁止静默降级。
- 证据：PyPI resolver 列出的最高版本为 `0.0.116`；官方 `master` 与 `v0.0.116` 均对应不可变提交 `f59781235353d57911df51fbf0d8d45683c13c0c`；仓库没有 lock、constraints 或 Git source override。使用 `0.0.116` 的独立兼容实验中，`pip check` 也准确保留了唯一的 floor 冲突。
- 适用范围：所有依赖未锁定、依赖声明领先于正式发布，或需要从官方源重建隔离环境的验证任务。
- 后续行动：环境创建前逐项核验关键 floor 与官方发布状态；不可满足时记录来源和解析证据，将严格验证标记为阻塞；只有在独立环境、明确例外和非严格结论下才可运行兼容实验。
- Tik 复审：2026-07-28 Tik 最终复审确认。
- 更正记录：无。

### LESSON-20260728-004: parity 双侧必须从同一语义阶段开始

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 背景：upstream dataset 在生成输入特征前已执行去水、去氢和 atom completion，而 OFoldX exporter 最初直接读取 raw mmCIF，导致 `HOH` 阻塞且两侧样本边界不同。
- 经验：parity 双侧必须从同一语义阶段开始；upstream 若已经完成水/氢过滤和 atom completion，raw input adapter 必须显式、结构化且可追溯地复现该边界，不能通过补充 CCD component 改变样本内容。
- 证据：结构化 adapter 在不修改 source 的前提下重建 `atom_site`，保留其他 categories 与 assembly metadata，并记录 source/prepared SHA、过滤统计和 charge policy；处理后 `4ame` 与 `1zev` 两侧 counts 分别精确为 `66/483`、`36/758`，identity 均无 `HOH`、`DOD`、H 或 D。
- 适用范围：比较不同 loader、dataset pipeline 或 feature processor 时，任何一侧在 raw source 与 tensor 之间存在隐式筛选、补全或规范化的任务。
- 后续行动：在设计 parity harness 时先列出两侧处理阶段和样本成员边界；用结构化 adapter 复现必要边界并记录源与派生 provenance；未知 component 必须要求 pinned resolver revision，不联网猜测，也不通过补 component 绕过样本过滤。
- Tik 复审：2026-07-28 Tik 最终复审确认。
- 更正记录：无。

### LESSON-20260728-005: 跨实现字段必须按生成语义映射

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 背景：OFoldX `token_to_rep_atom` 与 upstream 的 PAE、distogram representative masks 具有相近名称和相同 atom-axis shape，但生成用途不同。
- 经验：跨实现字段必须沿生成路径和下游用途确认语义对应关系，不能仅凭名称或 shape 建立映射。
- 证据：synthetic test 将 upstream PAE 与 distogram masks 故意设为不同值；语义修正后，真实 `4ame`、`1zev` 的 distogram representative 均精确一致，而 PAE representative 正确保留为 upstream-only expected difference。
- 适用范围：不同模型实现间的 feature schema 对齐，尤其是多个 mask、index 或 embedding 具有相同维度但服务于不同 head/operation 的情况。
- 后续行动：每个 mapped field 记录生产者、消费者、轴语义和数值约束；使用能让候选字段产生不同值的测试证明映射不是形状巧合。
- Tik 复审：2026-07-28 Tik 最终复审确认。
- 更正记录：无。

### LESSON-20260728-006: reference geometry 必须尊重 space partition 与双侧 mask

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 背景：reference coordinates 的每个 `ref_space_uid` 可拥有独立刚体坐标系，且两侧 reference mask 可能不同；全局 Kabsch 或单侧 mask 会混合不相关自由度。
- 经验：reference coordinate 诊断必须先取两侧有效 mask 交集，再按 `ref_space_uid` 等价分区逐组消除旋转和平移，并同时报告逐组 RMSD 与组内 pair-distance 误差。
- 证据：两个 synthetic space 分别施加不同旋转和平移后，逐空间诊断得到接近零的 weighted/max RMSD 和 pair-distance error；真实两样本则在正确分区后仍有稳定的非零几何差异。
- 适用范围：component-local reference conformer、模板局部坐标、刚体 frame 或任何允许分组独立坐标系的几何 parity。
- 后续行动：几何报告明确 partition、双 mask policy、group/valid counts、weighted/max RMSD 和 pair-distance 统计；禁止将所有空间合并后用单一刚体拟合得出结论。
- Tik 复审：2026-07-28 Tik 最终复审确认。
- 更正记录：无。

### LESSON-20260728-007: MSA 深度必须绑定真实来源与 query-row policy

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix real-MSA input tensor parity 验证
- 背景：旧 MSA-disabled 验证中两侧 `msa` 第一维均为 1，但该行只是 query-only fallback；真实 `4ame` MSA 又暴露出两侧对 query 行的自然组装策略不同。
- 经验：MSA 第一维为 1 通常只说明 query-only fallback，不能作为 real-MSA parity。真实验证必须记录 mapping、A3M path/SHA、raw/projected/deduplicated/effective/emitted depth、entity binding 和 query-row policy，并在缺失来源时禁止 silent fallback。
- 证据：`4ame` 的 mmseqs source 为 47 行且 row 0 是 query；Protenix 额外拼接 synthetic paired query 后为 `[48,66]`，OFoldX bare MSA 为 `[47,66]`。两侧 statistics exact，Protenix `[1:]` 与 OFoldX rows 一致，确认差异来自 query-row policy。`1zev` 所需的 `rna_msa/rna_sequence_to_pdb_chains.json` 与 `rna_msa/msas` 均缺失；本次选择的 base `ProtenixProcessor._normalize_msa` 只接受 protein MSA，因此标为 `REAL_MSA_BLOCKED`，没有把 `[1,36]` 记为成功。可接受 RNA 的 template-capable processor 属于另一个实验 surface。
- 适用范围：所有使用 MSA、模板搜索结果或其他可静默退化为单 query 输入的跨实现 parity 验证。
- 后续行动：将 source/depth/binding/query policy 作为 manifest 必填项；real-MSA 深度小于等于 1 或来源缺失时阻塞；仅把 suffix 对齐作为诊断，不得裁剪主导出以制造 shape pass。
- Tik 复审：2026-07-28 Tik 最终复审确认该原则与实现证据。
- 更正记录：无。

### LESSON-20260729-008: 归档成员校验与展示顺序应分离

- 日期：2026-07-29
- 来源任务：4ame Protenix/OFoldX input tensor notebook 验证
- 背景：NPZ 的成员顺序用于 notebook 全量展示，而 manifest 的 `arrays` 同时用于校验导出字段完整性；两种工件包含相同字段，但不保证 key order 相同。
- 经验：manifest `arrays` 只承担完整性契约，应使用 `len + set` 核对；展示和遍历必须以 NPZ `archive.files` 为准，并在加载时保持其插入顺序，不能断言 manifest key order 等于 NPZ order。
- 证据：Protenix 的 43 项和 OFoldX 的 30 项在 NPZ 与各自 manifest 中集合相同，但顺序均不同；首个差异分别为 Protenix `token_index` vs `asym_id`、OFoldX `token_mask` vs `atom_contact`。
- 适用范围：同时使用 NPZ 与 manifest 等双工件进行成员完整性检查和顺序敏感展示的任务。
- 后续行动：完整性与顺序使用不同断言；以 `len + set` 验证 manifest 契约，以 `archive.files` 驱动有序加载、遍历与输出。
- Tik 复审：2026-07-29 Tik 已确认条目内容与证据。
- 更正记录：无。

### LESSON-20260729-009: 大型 tensor dump 需要结构化验收

- 日期：2026-07-29
- 来源任务：4ame Protenix/OFoldX input tensor notebook 验证
- 背景：大型 ndarray 即使所在 cell 执行成功，也可能因展示截断、字段漏打或执行状态不连续而不能形成可信的完整 tensor dump。
- 经验：不能只检查 cell 执行成功；应解析每个 `field/shape/dtype/value` 块，核对字段块数量、NPZ order、shape/dtype、value 非空和无 `...`，再把输出字符规模作为截断异常的辅助信号，并检查 execution count 连续且无 error output。
- 证据：Protenix dump 为 43/43 blocks、`1,172,776` chars，OFoldX dump 为 30/30 blocks、`1,220,337` chars；两侧均 exact 遵循 NPZ order，shape/dtype exact、value 全部非空且无 ellipsis。9 个代码单元 execution count 为 1..9，且没有 error output。
- 适用范围：notebook 或日志中需要完整展示大型 tensor/ndarray，而非仅展示摘要的验证任务。
- 后续行动：对 dump 输出执行结构化 block parser 验收，并将字段顺序、元数据、非空 value、ellipsis、字符规模、连续 execution count 和 error output 一并纳入完成条件。
- Tik 复审：2026-07-29 Tik 已确认条目内容与证据。
- 更正记录：无。

### LESSON-20260729-010: exporter 前必须做 side-specific import/API gate

- 日期：2026-07-29
- 来源任务：4ame current-code Protenix/OFoldX fresh parity revalidation
- 背景：harness 的纯比较 helper self-test 在错误的 upstream 解释器中仍为 26/26，但真实 exporter 随后因 Biotite API 不兼容而在 import 阶段失败；第一次修订 gate 又因检查 parent package 而误判正确环境。
- 经验：helper self-test 不能证明真实 exporter 依赖可导入。每一侧在 export 前都必须使用实际解释器执行版本、模块路径和目标 API gate；API 符号还必须按照目标源码的实际 import namespace 检查，不能假设它会由 parent package re-export。
- 证据：`ofantibody` 解释器触发两个 upstream `failure.json`；pinned upstream venv 满足 `biotite==1.4.0` 等版本且 `BaseSingleDataset` 可导入。`PDBX_BOND_TYPE_ID_TO_TYPE` 位于 `biotite.structure.io.pdbx.convert`，不位于 parent `pdbx`；修正后的 R2 V2 gate、双侧 diagnose/self-test 和四个 exports 全部通过。
- 适用范围：任何跨环境 data exporter、插件加载器或运行时依赖只在真实入口导入的验证任务。
- 后续行动：执行顺序固定为 side-specific version/import/API/path gate、diagnose、self-test、真实 export；gate 的每个符号都从目标源码 import statement 反向核对 namespace，并将无效 gate 与真实环境 blocker 分开分类。
- Tik 复审：等待最终只读复审。
- 更正记录：无。

### LESSON-20260729-011: 多链样本不等于有效的 MSA pairing 样本

- 日期：2026-07-29
- 来源任务：3x24 Protenix/OFoldX paired-MSA parity 验证
- 背景：多链结构可能是同一序列的拷贝，也可能没有共享 species；此类样本即使 MSA depth 大于 1，也不能证明跨 entity pairing 模块被执行。
- 经验：paired-MSA case 必须同时满足至少两个不同 protein sequences、至少两个 MSA columns、共享非空 species、`paired_depth > 1`，并证明非 query paired rows 覆盖不同实体。
- 证据：3x24 的两个 canonical protein sequences 长度为 207/212，映射到两个独立 MSA 目录，具有 671 个共享 species；Protenix 输出 1332 个 paired rows，其中 1331 个非 query paired rows 均覆盖两个实体。
- 适用范围：验证 AlphaFold/Protenix 类 pipeline 的 multimer MSA pairing、species matching 或 paired/unpaired row assembly。
- 后续行动：case selection 阶段固定检查 sequence distinctness、source columns、shared species、paired counters 和每行 entity coverage；任一条件不满足时不得称为 pairing 验证。
- Tik 复审：2026-07-29 Tik 最终只读复审 `APPROVED`；case 的 distinct sequence、shared species、paired coverage 与计数均独立复算一致。
- 更正记录：无。

### LESSON-20260729-012: 外部追踪的 pairing plan 只能证明 lowering 一致

- 日期：2026-07-29
- 来源任务：3x24 Protenix/OFoldX paired-MSA parity 验证
- 背景：OFoldX raw reader 丢失 header/species，但 `MSACollection` 能消费显式 `row_pairing`/`row_is_paired`；如果不区分边界，lowering MATCH 容易被误写为端到端 pairing MATCH。
- 经验：外部按 Protenix 语义重建并审计的 pairing plan 可以验证 `MSACollection` 的行组装和特征 lowering，但不能证明 OFoldX 自身完成了 raw A3M species pairing。
- 证据：追踪计划的 `row_pairing`、`row_is_paired`、assembled MSA、raw deletion counts 与 Protenix engine exact；OFoldX 六字段在 canonical protein axis MATCH。报告仍将 `RAW_A3M_END_TO_END_PAIRING=NOT_IMPLEMENTED` 与 `TRACED_MSA_COLLECTION_LOWERING=MATCH` 分开。
- 适用范围：任何通过外部 adapter、计划文件或预计算索引驱动被测实现下游模块的 parity 验证。
- 后续行动：报告必须明确原始输入边界、外部注入内容和被实际验证的最低层合同；禁止把下游 lowering 结论外推为上游 parsing/pairing 结论。
- Tik 复审：2026-07-29 Tik 最终只读复审 `APPROVED`；lowering MATCH 与 raw A3M `NOT_IMPLEMENTED` 的范围边界表达准确。
- 更正记录：无。

### LESSON-20260729-013: parity artifact 必须绑定生成 harness

- 日期：2026-07-29
- 来源任务：3x24 Protenix/OFoldX paired-MSA parity 验证
- 背景：环境、源码 commit 和输入文件均已固定，但旧 artifacts 没有记录生成它们的 harness path/SHA；后续修改 harness 后再 compare 会形成不可审计的代码与结果错配。
- 经验：跨进程 parity 的每份 manifest 都必须保存生成 harness 的绝对路径和内容 SHA256，compare 还必须把缺失、路径差异和哈希漂移视为整体失败；不能只依赖当前工作树状态或最终脚本哈希。
- 证据：新增 focused tests 覆盖保存、缺失和漂移；全新 `r2` 四份 manifest 的 harness path/SHA 唯一值数量为 1，report 分类为 `MATCH`。旧目录明确标记为 `SUPERSEDED_FAILED_PROVENANCE_ATTEMPT`。
- 适用范围：任何由多个解释器、环境、repeat 或阶段分别生成 artifacts，最后再离线汇总的 parity/reproducibility 验证。
- 后续行动：在第一次 export 前冻结 harness；每个 writer 自动记录自身 path/SHA；aggregate 加载所有 manifests 后先做 provenance gate，再进行数值结论。
- Tik 复审：2026-07-29 Tik 最终只读复审 `APPROVED`；四份 manifest 与当前 harness 的绝对路径/SHA 一致，缺失和漂移的失败测试通过。
- 更正记录：无。

### LESSON-20260729-014: raw dict verdict 与 identity-projected 字段 verdict 必须分层

- 日期：2026-07-29
- 来源任务：3x24 Protenix/OFoldX paired-MSA 全字段 parity 验证
- 背景：Protenix 与 OFoldX 的原始字段集合、token/atom 轴和 modified-residue 表示不同；只给一个 match/mismatch 会把“完整输入对象不等”与“共同语义字段可比”混为一谈。
- 经验：全字段 parity 应先给 raw dict 的整体结论，再在明确且可审计的 canonical identity projection 上逐字段分类。允许的 normalization 必须绑定语义合同；不能为了提高 match 数量而归一化仍有模型意义的 within-token atom index 或 representative atom 选择。仅证明每个 root 名称出现在某行还不够；每个 raw value 必须绑定真实 row/gate selector，复合字段必须以互斥且无遗漏的 partition 覆盖完整 channel 轴。
- 证据：3x24 raw dict 为 43 vs 30 fields、441 vs 419 tokens、3287 vs 3275 atoms，因此整体 `NOT_EQUAL_AS_A_WHOLE`；按 `(entity_id, monomer_index, canonical_component[, atom_name])` 投影后得到 419/3275 公共轴。R3 复审发现常量 `token_index`、错读 packed mask 和遗漏 `token_features[:,0:384]` 可在 root-name coverage 下漏过；R4 用全 raw token axis gate、duplicate mask gate、prefix-zero gate和复合 selector audit 修复后，P43/O30 consumption audit、全部 fixed gates、双 repeat 与外部 37,030,861-value parser 均通过。
- 适用范围：不同实现具有字段打包、索引编号、modified component 展开、派生字段或缺省 feature 差异的 model-input parity。
- 后续行动：建立完备 raw field registry；为每行同时记录 raw 与 compared metadata、projection/derivation 方法和结论；为每个 root 记录实际 consumer、selector、method、status，并用 gap/overlap/out-of-range、selected/dropped value mutation 和 declared-but-unread 负向测试锁定；把 raw-object verdict、投影字段 verdict、未实现的上游边界分别展示。
- Tik 复审：2026-07-29 Tik R4 最终独立只读复审 `APPROVED`；确认 raw value selector/gate audit、负向测试和最终证据满足该经验。
- 更正记录：无。

### LESSON-20260729-015: loss parity 必须由双侧 native training contract 解锁

- 日期：2026-07-29
- 来源任务：Protenix/OFoldX same-output native loss parity 验证
- 背景：旧 loss R1 将 synthetic/common contract 或 upstream supplement 送入两套 loss 公式，能诊断公式差异，却没有证明两个 production data pipeline 会产生相同 native input/target。
- 经验：same-output native loss parity 只能在双侧 raw-to-training pipeline 成功、原始 instance/entity/component/atom identity 构成完整双射、target coordinate/mask/index 的 shape/dtype/value exact、output head 的 axes/shape/dtype/bin domain exact 后执行。Projected lowering、prepared input、canonical collapse、字段补齐和 synthetic/common contract 均不能解锁该 gate；无 counterpart 必须保留缺失状态，禁止用零填充制造可计算结果。
- 证据：fresh native pipeline R6 中 Protenix 三个 case 双 repeat 均通过官方 `run_gen_data -> two-pass parse_configs -> get_datasets -> get_dataloaders`，模型专用 merge 将 `N_cycle` 从 4 更新为 10，并保存完整 numeric input/label/full tensors、config/source/payload/MSA provenance；OFoldX 的 `3x24`/`4ame` 在默认 raw reader 分别因 entity definition/formal charge 冲突失败，`1zev` reader PASS 后因 `HOH` reference coordinates 在 processor 失败。六组 repeat determinism 均 MATCH，所有记录的 status/index/A3M/path/SHA 和独立 production-bin probe 被 finalizer 与 loss loader 递归复算。fresh loss R7 因此报告 `SAME_OUTPUT_NATIVE_LOSS=NOT_TESTABLE` 和 `FULL_OBJECTIVE_MEMBERSHIP=MISMATCH`，未执行 eligible-case loss consumer。
- 适用范围：不同实现之间的 loss、metric、objective 或训练 step parity，尤其当数据 schema、identity axis 或 target 构造不一致时。
- 后续行动：先保存双侧 native stage evidence 和完整 identity/target hashes；只有所有 gate 通过才生成唯一 prediction artifact 并交给两个 pinned-environment consumer。任一 gate 失败时直接结束数值比较并明确 `NOT_TESTABLE`。
- Tik 复审：等待最终只读复审。
- 更正记录：2026-07-29 更正旧记录中“OFoldX 三例均在 raw reader 因 CSD/HOH 失败”的错误；`1zev` 实际在 processor 阶段失败。

### LESSON-20260729-016: consumer 可执行性与 parity 结论必须分离

- 日期：2026-07-29
- 来源任务：Protenix/OFoldX same-output native loss parity 验证
- 背景：没有 dual native success 时，正式 loss consumer 不应执行；但仅靠静态源码检查又不能证明 eligible path 绑定的 production API 实际可调用。
- 经验：应使用明确标记为 `INTERFACE_SMOKE_ONLY_NOT_PARITY_EVIDENCE` 的最小非退化 fixture 分别调用 production consumer，并额外执行真实 Processor -> Model -> common artifact -> consumer 路径；如果缺少另一侧同 fixture 的 native payload，E2E 本身必须以明确 FAIL 结束。同时让正式 parity 仍受 native payload、identity、target、head schema 和 prediction hash gate 控制。
- 证据：OFoldX smoke 通过 `MODELS.lookup_model('protenix_template').criterion` 调用注册的 `denoising_criterion_factory`；Protenix smoke 调用 `protenix.model.loss.ProtenixLoss`。真实 OFoldX E2E 的 `ProtenixTemplateProcessor`、tiny `ProtenixTemplateModel.forward`、identity-only common artifact 和落盘后新进程 criterion 均通过，但整体为 `FAIL_NO_NATIVE_PROTENIX_FIXTURE_PAYLOAD`。fresh R6/R7 因没有 eligible case 仍保持正式 consumer 未执行、`NOT_TESTABLE`；21/21 tests 锁定接口可执行性与 parity fail-closed 边界。
- 适用范围：production API 需要先证明可达，但真实比较样本被上游 admission gate 阻断的 loss、metric 或训练-step parity。
- 后续行动：所有 smoke artifact 必须包含 production symbol/source SHA 和非 parity 标签；正式报告只从 eligible-case consumer outputs 读取数值，objective membership mismatch 时永不允许 PASS。
- Tik 复审：等待最终只读复审。
- 更正记录：无。

### LESSON-20260729-017: model/processor/config/criterion 必须来自同一 registry 条目

- 日期：2026-07-29
- 来源任务：Protenix/OFoldX same-output native loss parity 第四轮验证
- 背景：released v1 variant 返回 `ProtenixTemplateModelConfig`；如果把它传给 base `ProtenixModel` 或用 base `ProtenixProcessor`，构造表面相近但 template graph 与 feature producer 不属于同一 architecture contract。reference-position augmentation 的默认值也与 Protenix data config 不同。
- 经验：生产 parity 不能分别挑选 model、processor、config 和 criterion。必须从同一个 registry ID 获取完整四元组，并显式对齐所有影响数值的 processor config；本例使用 `protenix_template` 与 `ProtenixProcessorConfig(augment_reference_positions=False)`。
- 证据：registry construction/call tests 验证实际类分别为 `ProtenixTemplateModel`、`ProtenixTemplateProcessor`、`ProtenixTemplateModelConfig` 和 `denoising_criterion_factory`；tiny production forward/criterion 通过，双侧 preprocessing artifact 均记录 effective augmentation=false。
- 适用范围：variant 会改变模型子图、processor feature 集或 criterion factory 的所有 registry-driven 训练/推理框架。
- 后续行动：artifact 固定记录 architecture ID、四个实际类/工厂、config 类型与关键 processor 参数；任意 registry 交叉组合或默认参数漂移必须 fail closed。
- Tik 复审：等待复审。
- 更正记录：无。

### LESSON-20260729-018: 保存后重载和边界点测试是 loss 证据的一部分

- 日期：2026-07-29
- 来源任务：Protenix/OFoldX same-output native loss parity 第五轮验证
- 背景：producer 在内存中直接调用 criterion，只能证明同一进程对象可用，不能证明正式 consumer 实际读取了已声明的 NPZ/manifest；用通用等距边界描述 head 也会漏掉严格 `>`、float32 rounding、PAE 平方、clamp 和 one-hot offset。
- 经验：E2E producer 必须只生产并持久化数据，consumer 必须在新进程中重载 manifest/NPZ 后调用注册 criterion，并回写实际消费的 artifact SHA。离散 head 契约必须在两个 pinned 环境中分别直接调用各自 production decoder；Protenix label 必须直接调用真实 loss method，几何值注入可以 patch，但 classification/one-hot 分支不得 patch。共享公式重建不能作为双侧观测。
- 证据：R7 E2E 的 producer 明确记录 `criterion_executed=false`，新 OFoldX 进程 consumed SHA `37758f69...f82192` 与持久化 common NPZ exact；预测 NPZ和 manifest 篡改均被 formal consumer 拒绝。R6 独立调用 Protenix `get_bin_centers` 与 OFoldX `_protenix_bin_centers`，记录不同 callable path/SHA 后四个 head 输出 exact；真实 `DistogramLoss.calculate_label`、`PLDDTLoss.bins_from_lddt`、`PDELoss.calculate_label`、`PAELoss.calculate_label` 在 float32 boundary/nextafter/endpoints/clamp/-1 offset 上均得到预期 argmax，PAE 使用 squared-error domain；21/21 focused tests PASS。
- 适用范围：跨进程 loss/metric parity、缓存重放、离散分类 head、任何以 manifest 声明 artifact 身份的审计。
- 后续行动：把 producer 与 consumer 分成独立命令；正式记录同时包含 declared SHA 和 consumer-observed SHA；bin 测试不得只抽查内部点、用 float64 重建 production float32 tensor，或让两侧共享同一 reconstruction helper。
- Tik 复审：2026-07-29 第五轮只读复审为 `REVISE`，指出共享公式重建不能证明两侧 production callable；R6/R7 已用独立 pinned subprocess 直接调用两侧 decoder 和四个真实 Protenix label method 修复。第六轮指出 notebook 依赖外部预注入 `display`；当前已显式导入并以 clean globals 6/6 顺序执行，等待第六轮简审批准。
- 更正记录：第五轮 direct-production-call blocker 与第六轮 notebook clean-globals blocker 均已按复审意见修复，未改变 `NOT_TESTABLE` 结论。

### LESSON-20260730-019: model input parity 必须在真实 runner forward 边界观测

- 日期：2026-07-30
- 来源任务：Protenix/OFoldX runner-boundary input 与 loss parity 验证
- 背景：直接调用 reader、processor、collator 或把外部 pairing plan 交给 `MSACollection`，可以诊断局部 lowering，却不能证明训练 Runner 实际加载并送入模型的输入相同。
- 经验：runner input parity 应在真实 one-epoch 入口注册 pass-through forward pre-hook。hook 只递归 detach/clone CPU 数据并返回 `None`，不得 sentinel 中断；任一 epoch 失败时 capture 都不可采纳，并立即停止该 case。Runner 缺失的 ingestion 能力必须记录为产品边界，不能用 harness 注入对象或 path 补齐。
- 证据：旧 R1 使用 stale compat wheel，现已标 `STALE_WHEEL_INVALID`。current-code R2 从 clean tracked source 构建 wheel 并 fresh install，wheel/direct_url/SHA 可审计，但生成包缺少 `ofoldx._version`，四次 OF worker 均在 import 边界 fail closed；Protenix official runner 未启动，输入和 `<1e-7` loss 仍为 `NOT_TESTABLE`。
- 适用范围：跨实现训练输入、loss、metric 或任何需要证明 production data-to-model 边界一致的验证。
- 后续行动：先修复 raw reader 和 Runner MSA ingestion；之后以相同 raw SHA、双 repeat、双 completed epoch、hook captures、identity/target gate 重新验证。旧 adapted/projected evidence 永久保留 diagnostic-only 标签。
- Tik 复审：等待复审。
- 更正记录：2026-07-30 撤回 R1 作为 current-code runner 证据；R2 暴露更早的 current wheel packaging blocker，因此尚不能对当前 Runner MSA ingestion 作正向或负向运行时结论。

## Entry Template

<!--
### LESSON-YYYYMMDD-NNN: 简短标题

- 日期：
- 来源任务：
- 背景：
- 经验：
- 证据：
- 适用范围：
- 后续行动：
- Tik 复审：
- 更正记录：
-->
