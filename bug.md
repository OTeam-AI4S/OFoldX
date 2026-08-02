# TikTok Bug Log

## 用途

记录 TikTok 流程中实际发现的代码缺陷、回归、验证失败和流程故障，并保留从发现到关闭的证据链。

## 状态

- `Open`：缺陷已确认，尚未修复。
- `Investigating`：现象存在，但原因或缺陷边界尚未确认。
- `Fixed`：Tok 已实施修复并提供验证证据，等待或已完成 Tik 复审。
- `Deferred`：缺陷已确认，但经明确决定不在当前范围修复。
- `Closed`：Tik 已确认修复和验证满足验收条件。

## 记录规则

- 条目 ID：`BUG-YYYYMMDD-NNN`
- Tik 确认描述、严重级别和验收条件，Tok 写入或更新。
- 严重级别使用 `Critical`、`High`、`Medium` 或 `Low`。
- `Fixed` 必须记录修复文件和验证证据。
- 只有 Tik 可以在复审结论中授权改为 `Closed`。
- 不删除历史条目；状态变化追加到 History。

## Active Bugs

### BUG-20260728-001: compare 混淆工件阻塞与 parity 失败

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Closed
- 严重级别：High
- 影响：identity/schema/core field mismatch 被错误报告为 `BLOCKED`，diagnose 可能跨侧误报依赖，`has_deletion` 未强制 exact，旧 self-test 未经过真实 compare 路径。
- 复现步骤：加载语法有效但 identity 或 core field 不一致的双侧 artifact 后运行 `compare`。
- 预期行为：仅缺失或损坏导出 artifact 返回 `BLOCKED`/exit 2；加载后的 identity、axis、schema、shape、dtype、value mismatch 返回 `FAIL`/exit 1。
- 实际行为：加载与对齐共用宽泛异常分支，首个异常直接返回 `BLOCKED`/exit 2。
- 根因：artifact I/O 与 parity evaluation 没有分层，测试只断言了手写记录。
- 修复位置：`scripts/parity/protenix_data_parity.py` 的四工件加载聚合、parity evaluation 分支、分侧 diagnose、exact field 集合和真实 self-test。
- 验证证据：`self-test` 13/13 通过；`test_identity_mismatch_is_fail_exit_one`、`test_loaded_core_schema_axis_shape_and_dtype_mismatches_exit_one`、`test_real_missing_and_extra_field_classification`、`test_has_deletion_is_exact` 均经过真实 `compare()` 与 JSON report；两份现场 compare 报告各聚合 4 条 artifact blocker 并 exit 2。
- 验收条件：仅加载失败 exit 2；加载后 core/identity/axis/value mismatch 均 FAIL/exit 1；真实路径测试覆盖。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 第二轮复审通过，授权 Closed。

#### History

- 2026-07-28：状态记录为 `Open`，Tik 第二轮审查确认缺陷与验收条件。
- 2026-07-28：状态由 `Open` 更新为 `Fixed`；加载失败与 parity mismatch 已分层，真实路径 self-test 通过，等待 Tik 批准 `Closed`。
- 2026-07-28：状态由 `Fixed` 更新为 `Closed`；Tik 第二轮复审通过，self-test 13/13、synthetic compare 路径与 blocker 聚合验证满足验收条件。

### BUG-20260728-002: token identity 与 residue tensor 语义错误

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Closed
- 严重级别：High
- 影响：标准聚合 token 使用不同 representative atom 建 key 会产生伪 identity mismatch；真实 `residue_index` 污染会被 identity 派生值覆盖。
- 复现步骤：让 baseline centre atom 与 OFoldX distogram representative 不同，或仅污染一侧 `residue_index` tensor。
- 预期行为：聚合 token 只按 chain、链内规范 residue、component 对齐；逐原子 token 才增加 atom name；真实 residue tensor 链内减 min 后比较。
- 实际行为：token key 强制包含 representative atom，且 baseline identity residue 替代了真实 tensor。
- 根因：identity 轴对齐与特征值比较边界混淆。
- 修复位置：`scripts/parity/protenix_data_parity.py` 的 `_baseline_identity`、`_ofoldx_identity`、`_axis_permutation`、`_normalize_residue_tensor` 与 `_align_and_transform`。
- 验证证据：`test_aggregate_token_identity_ignores_representative_metadata` 与 `test_residue_tensor_pollution_is_fail_exit_one` 通过；污染报告保留 baseline raw `[10, 11]`、OFoldX raw `[20, 22]` 并将 `residue_index` 分类为 `FAIL`。
- 验收条件：标准 token 不以 representative 命名，逐原子 token 才使用 atom name，真实 residue tensor 按链规范化，污染必须 FAIL。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 第二轮复审通过，授权 Closed。

#### History

- 2026-07-28：状态记录为 `Open`，Tik 第二轮审查确认缺陷与验收条件。
- 2026-07-28：状态由 `Open` 更新为 `Fixed`；identity 与 tensor 值边界已分离，等待 Tik 批准 `Closed`。
- 2026-07-28：状态由 `Fixed` 更新为 `Closed`；Tik 第二轮复审通过，聚合 token identity 与 residue tensor 污染 synthetic 测试满足验收条件。

### BUG-20260728-003: export 失败可遗留 stale 成功工件

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Closed
- 严重级别：High
- 影响：失败重跑后旧 arrays/manifest/identity 仍可能被 compare 当作成功结果读取。
- 复现步骤：先成功写入目标 repeat 目录，再让同目录 export 失败。
- 预期行为：每次 export 开始清理目标目录；失败目录只保留 failure，成功目录无 stale failure。
- 实际行为：export 与 failure writer 都未清理互斥工件。
- 根因：artifact 生命周期缺少目录级 reset。
- 修复位置：`scripts/parity/protenix_data_parity.py` 的 `_reset_artifact_dir`、`_save_artifact`、`_write_failure` 及两个 export 入口。
- 验证证据：`test_stale_artifacts_are_mutually_exclusive` 通过；现场重跑 8 个 export 后 failure 数量为 8，arrays/manifest/identity 数量为 0，每个 repeat 目录只含 `failure.json`。
- 验收条件：失败目录仅 failure，成功目录没有 stale failure。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 第二轮复审通过，授权 Closed。

#### History

- 2026-07-28：状态记录为 `Open`，Tik 第二轮审查确认缺陷与验收条件。
- 2026-07-28：状态由 `Open` 更新为 `Fixed`；失败与成功工件已互斥，等待 Tik 批准 `Closed`。
- 2026-07-28：状态由 `Fixed` 更新为 `Closed`；Tik 第二轮复审通过，stale artifact self-test 及现场 8 failure、0 success artifact 验证满足验收条件。

### BUG-20260728-004: parity harness 被 gitignore 隐藏

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Closed
- 严重级别：Medium
- 影响：验证脚本不会出现在普通 Git 状态和交付 diff 中。
- 复现步骤：运行 `git check-ignore scripts/parity/protenix_data_parity.py`。
- 预期行为：目标脚本不被忽略，`scripts/parity` 的其他本地工件继续忽略。
- 实际行为：`.gitignore` 忽略整个 `scripts/parity/`。
- 根因：忽略规则粒度过粗。
- 修复位置：`.gitignore` 的 `scripts/parity/*` 与目标脚本反向规则。
- 验证证据：`git check-ignore scripts/parity/protenix_data_parity.py` 无输出且 exit 1；`git status --short --untracked-files=all` 显示 `?? scripts/parity/protenix_data_parity.py`，其他 parity 工件仍被忽略。
- 验收条件：`git check-ignore` 不命中且 `git status --short` 显示脚本。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 第二轮复审通过，授权 Closed。

#### History

- 2026-07-28：状态记录为 `Open`，Tik 第二轮审查确认缺陷与验收条件。
- 2026-07-28：状态由 `Open` 更新为 `Fixed`；目标 harness 已从忽略规则中放行，等待 Tik 批准 `Closed`。
- 2026-07-28：状态由 `Fixed` 更新为 `Closed`；Tik 第二轮复审通过，目标脚本 check-ignore exit 1、Git 状态可见且受保护范围无差异。

### BUG-20260728-005: 本地缺少可完成双侧导出的环境

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Closed
- 严重级别：High
- 影响：当前无法产生两侧 tensor artifact，因此无法给出最终 parity 结论。
- 复现步骤：在 `ofantibody` 环境分别运行 upstream 与 OFoldX export。
- 预期行为：两侧分别成功导出 `4ame`、`1zev` 的两个 repeat。
- 实际行为：baseline 被 Biotite 1.0.1 API 不兼容阻断；OFoldX 缺 `_version.py`；环境缺 tensordict。
- 根因：当前可用环境与两个源码树的运行要求不匹配。
- 修复位置：本轮禁止修改依赖、环境、`ofoldx/` 或 `baseline/Protenix`，因此不修复。
- 验证证据：`/tmp/ofoldx-protenix-parity/diagnostics/` 与各 repeat `failure.json`。
- 验收条件：独立环境成功导出两样本双重复。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 最终复审确认后续独立环境的 8/8 导出满足验收条件，授权 `Closed`。

#### History

- 2026-07-28：状态记录为 `Deferred`，受本轮禁止安装和 patch 的范围约束。
- 2026-07-28：状态由 `Deferred` 更新为 `Closed`；后续独立环境完成两样本双侧双重复 8/8 成功导出，原环境阻塞描述已被后续证据 superseded，验收条件满足。

### BUG-20260728-006: OFoldX 声明未发布的 chanfig 版本 floor

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Fixed
- 严重级别：High
- 影响：从官方源创建 OFoldX clean install 时依赖解析不可满足，严格 parity 环境无法进入 `pip check` 和 API gate。
- 复现步骤：在新建虚拟环境中从固定 OFoldX 提交安装包，并从官方 PyPI 解析其声明的 `chanfig>=0.0.117`。
- 预期行为：项目声明的最低版本存在官方可获取发布，clean install 可解析并通过 `pip check` 与目标 API gate。
- 实际行为：官方 PyPI 最高仅有 `chanfig==0.0.116`；官方仓库 `master` 与 `v0.0.116` 也停留在同一 `0.0.116` 提交，严格解析失败。
- 根因：OFoldX 的依赖 floor `chanfig>=0.0.117` 高于 CHANfiG 当前官方发布版本，且项目没有提供 lock、constraints 或官方 Git source override。
- 修复位置：尚未修复；需要 CHANfiG 发布满足 floor 的官方版本，或由 OFoldX 源码修正其依赖声明。
- 验证证据：严格环境日志 `/tmp/ofoldx-parity-env-c52d761-c3bfc365/logs/ofoldx-runtime-deps-install.log`；兼容实验使用 `chanfig==0.0.116` 后，`/tmp/ofoldx-parity-env-compat116-c52d761/logs/pip-check.log` 仅保留该声明冲突。
- 验收条件：官方发布满足声明版本，或源码修正声明；随后从空环境完成 clean install，并通过零问题 `pip check` 和完整 API gate。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 指定保持 `Open`，等待实际修复与验收。

#### History

- 2026-07-28：状态记录为 `Open`；官方来源不存在满足 OFoldX 声明 floor 的 CHANfiG 版本。

### BUG-20260728-007: harness 未镜像 upstream solvent/source normalization

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Closed
- 严重级别：High
- 影响：OFoldX exporter 直接读取 raw mmCIF，保留了 upstream 已删除的 solvent，并从不同语义样本边界开始比较；`4ame` 因未知 `HOH` component 阻塞。
- 复现步骤：在未预处理 raw mmCIF 的情况下，对 `/data/shared/datasets/protenix/mmcif/4ame.cif` 运行 `export-ofoldx`。
- 预期行为：OFoldX 侧显式复现 upstream 的去水、去氢和 completion 前输入边界，保留 assembly metadata，并记录可追溯的派生输入 provenance。
- 实际行为：原 harness 将 raw mmCIF 直接交给 OFoldX `read_structure`，遇到 `HOH` 后返回 exit 2。
- 根因：harness 只对齐了 processor 配置与 tensor 轴，没有对齐 upstream dataset 已隐式执行的 source normalization 阶段。
- 修复位置：`scripts/parity/protenix_data_parity.py` 的 `_prepare_ofoldx_mmcif()`、`export_ofoldx()` input preprocessing 与 synthetic self-tests。
- 验证证据：compat self-test 16/16；`4ame` gate 精确得到 `66/483`；完整 8 次导出中 `4ame` 与 `1zev` 两侧分别精确得到 `66/483`、`36/758`；全部 OFoldX identity 无 `HOH`、`DOD`、H 或 D，source/prepared hash 在重复间稳定。
- 验收条件：synthetic tests 覆盖 solvent/H/D 过滤、源文件不变、确定性统计/hash 和 retained component 集合；两样本 counts 正确且 identity 无 solvent。
- Owner：Tok
- Tik 复审：2026-07-28 Tik REVISE 复审通过，授权 `Closed`。

#### History

- 2026-07-28：状态记录为 `Open`；真实 `4ame` export 被 raw mmCIF 中的 `HOH` 阻塞。
- 2026-07-28：状态由 `Open` 更新为 `Fixed`；结构化 adapter、自测、两样本正确 counts 和无 solvent identity 满足实现侧验收，等待 Tik 关闭。
- 2026-07-28：状态由 `Fixed` 更新为 `Closed`；Tik 复审确认 synthetic coverage、两样本 counts、identity 和 provenance 满足验收条件。

### BUG-20260728-008: OFoldX mmCIF unknown formal charge 被物化为零

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Deferred
- 严重级别：High
- 影响：Biotite 读取 mmCIF 的 unknown `pdbx_formal_charge` 时可能将其物化为 `0`，从而与带正式电荷的 OFoldX built-in component（例如 ARG）冲突并阻断结构加载。
- 复现步骤：保留 atom_site 中全为 `?` 或 `.` 的 `pdbx_formal_charge` 列，经 Biotite/OFoldX production loader 读取并与带正式电荷的 built-in component 对照。
- 预期行为：mmCIF unknown formal charge 保持为 `None`，由已知 component chemistry 提供正式电荷，不应被解释为显式零电荷。
- 实际行为：unknown 值可能被解释为 `0`，导致 source atom 与 built-in component 的 charge 校验不一致。
- 根因：production mmCIF loader 在 Biotite mask 到 OFoldX optional formal charge 的转换中没有保留 unknown 语义。
- 修复位置：生产 loader 修复不在本轮范围；harness 仅对整列全 unknown 的情况无损删除临时 CIF 列，known/unknown 混合列明确阻塞。
- 验证证据：synthetic tests 覆盖全 unknown 列删除、全 known 保留和 mixed blocker；两样本 manifest 的 `charge_policy`、source/prepared SHA 和行统计保留该适配证据。
- 验收条件：production loader 对 `?`/`.` 保持 `None`，并通过包含 ARG 等正式带电 built-in component 的 regression test。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 指定本轮 `Deferred`，禁止修改 production loader。

#### History

- 2026-07-28：状态记录为 `Deferred`；本轮只允许 harness 级、整列全 unknown 的无损适配，生产修复不在范围内。

### BUG-20260728-009: representative atom 字段按名称误映射

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Closed
- 严重级别：High
- 影响：OFoldX `token_to_rep_atom` 被错误映射为 upstream `pae_rep_atom_mask`，产生伪 PAE mismatch，同时没有验证语义对应的 `distogram_rep_atom_mask`。
- 复现步骤：将 OFoldX `token_to_rep_atom` 展开为 atom mask 后，与 upstream `pae_rep_atom_mask` 比较。
- 预期行为：跨实现字段按生成语义映射；`token_to_rep_atom` 对应 upstream distogram representative，PAE representative 保持 upstream-only expected difference。
- 实际行为：原 harness 仅依据相似名称和 shape 映射为 PAE representative。
- 根因：字段映射没有追踪两侧 representative mask 的生成用途。
- 修复位置：`scripts/parity/protenix_data_parity.py` 的 `_align_and_transform()`、core field 集合和 representative semantic self-test。
- 验证证据：双环境 self-test 19/19；synthetic PAE 与 distogram mask 故意不同仍只让 distogram 进入 core compare；真实 `4ame`、`1zev` 的 `distogram_rep_atom_mask` 均 `PASS_EXACT`，`pae_rep_atom_mask` 为 upstream-only `EXPECTED_DIFFERENCE`。
- 验收条件：synthetic test 能区分 PAE 与 distogram representative；真实两样本比较使用 distogram mask 且 PAE 不产生伪失败。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 最终复审确认 semantic test 与真实 distogram evidence，授权 `Closed`。

#### History

- 2026-07-28：状态记录为 `Open`；Tik 复审确认 representative 字段语义映射错误。
- 2026-07-28：状态由 `Open` 更新为 `Fixed`；synthetic semantic test 与真实两样本 distogram exact evidence 通过，等待 Tik 关闭。
- 2026-07-28：状态由 `Fixed` 更新为 `Closed`；Tik 最终复审确认 representative 语义映射和验收证据满足条件。

### BUG-20260728-010: dtype mismatch 提前返回隐藏 value 结论

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Closed
- 严重级别：High
- 影响：字段 dtype 不同时 compare 立即返回，无法区分“仅 dtype 不同”与“dtype 和数值都不同”，掩盖真实数据一致性边界。
- 复现步骤：比较 shape 相同但 dtype 不同的数组，分别令数值相同和数值不同。
- 预期行为：dtype 仍按 strict contract 判定 `FAIL`，同时在共同数值域继续报告 value classification、误差计数、示例和最大误差。
- 实际行为：原 `_compare_arrays()` 在 dtype mismatch 后直接返回，没有 value evidence。
- 根因：dtype contract 与 value diagnostics 共用一个提前返回分支。
- 修复位置：`scripts/parity/protenix_data_parity.py` 的 `_compare_arrays()` 与 dtype/value synthetic tests。
- 验证证据：双环境 self-test 19/19 覆盖 dtype 不同值同和值异；真实两样本中 `ref_atom_name_chars`、`ref_charge`、`ref_element` 为 strict dtype `FAIL` 但 value `PASS_EXACT`，`ref_mask` 与 `ref_mask_packed` 同时报告 dtype 和 value `FAIL` 及具体原子索引。
- 验收条件：dtype mismatch 永不被归一化为通过；shape 可比较时始终产生独立 value status 与误差细节。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 最终复审确认 dtype/value 拆分报告与覆盖，授权 `Closed`。

#### History

- 2026-07-28：状态记录为 `Open`；Tik 复审确认 dtype early return 丢失 value evidence。
- 2026-07-28：状态由 `Open` 更新为 `Fixed`；synthetic 双场景和真实 dtype/value 拆分报告通过，等待 Tik 关闭。
- 2026-07-28：状态由 `Fixed` 更新为 `Closed`；Tik 最终复审确认 strict dtype 与共同数值域诊断同时满足验收条件。

### BUG-20260728-011: reference coordinates 使用全局 Kabsch 诊断

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Closed
- 严重级别：High
- 影响：把不同 `ref_space_uid` 的 reference coordinates 合并做一次 Kabsch，会将各空间允许的独立刚体变换误报为几何差异，并且原诊断只使用 upstream mask。
- 复现步骤：构造两个 reference space，对每个 space 分别施加不同旋转和平移后运行旧全局 Kabsch。
- 预期行为：使用两侧有效 mask 交集，并按 `ref_space_uid` 等价分组逐空间对齐；报告 group count、valid atoms、加权/最大 RMSD 和组内 pair-distance 误差。
- 实际行为：原 harness 合并所有 reference atoms 做一次全局 Kabsch，仅接收一个 mask。
- 根因：reference coordinate diagnostic 忽略了 space partition 和双侧有效性语义。
- 修复位置：`scripts/parity/protenix_data_parity.py` 的 `_reference_geometry_diagnostic()`、parity report 和 multi-space self-test。
- 验证证据：双环境 self-test 19/19；两个 space 独立旋转/平移且两侧各排除一个 atom 时，逐空间 weighted/max RMSD 与 pair-distance max error 均接近零；真实报告按 66/36 个 space 分组并使用 482/756 个双 mask 有效 atoms。
- 验收条件：multi-space 独立刚体变换测试通过，几何报告包含逐组与聚合统计，且有效 atom 必须同时满足两侧 mask。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 最终复审确认逐 space、双 mask 几何诊断与覆盖，授权 `Closed`。

#### History

- 2026-07-28：状态记录为 `Open`；Tik 复审确认全局 Kabsch 与单侧 mask 诊断边界错误。
- 2026-07-28：状态由 `Open` 更新为 `Fixed`；逐 space、双 mask 几何实现及 synthetic/真实证据通过，等待 Tik 关闭。
- 2026-07-28：状态由 `Fixed` 更新为 `Closed`；Tik 最终复审确认 multi-space regression 与真实诊断满足验收条件。

### BUG-20260728-012: 真实 reference dtype、mask 与 geometry divergence

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix input tensor parity 验证
- 状态：Investigating
- 严重级别：High
- 影响：兼容实验中两样本的 reference tensor dict 均未达到 strict parity；差异涉及 dtype contract、有效 mask 和每个 component reference space 的内部几何。
- 复现步骤：加载 compat 输出根中 `4ame`、`1zev` 两侧双重复 artifacts 并运行 compare。
- 预期行为：映射后的 reference 字段 dtype、mask 值和按 `ref_space_uid` 消除刚体自由度后的内部几何一致。
- 实际行为：`ref_atom_name_chars`、`ref_charge`、`ref_element` 数值精确一致但 upstream 为 `int64`、OFoldX 为 `float32`；`ref_mask`/`ref_mask_packed` 还存在值差，`4ame` 为 A0 residue 66 `OXT` 一处，`1zev` 为 A0/B1 residue 1 `OP3` 两处；`ref_pos` dtype 相同但数值与逐 space 几何不同。
- 根因：尚未确认；可能涉及两侧 component reference-coordinate 数据源、terminal atom 定义、nucleic atom 定义及 feature dtype contract 的组合差异，需要分别追踪生成路径。
- 修复位置：调查中；本轮只改 harness 诊断，不修改 OFoldX 或 upstream production code。
- 验证证据：`4ame` 66 groups/482 valid atoms，weighted RMSD `0.925082`、max group RMSD `1.570546`、1608/1608 pair distances 超差、max pair error `3.573475`；`1zev` 36 groups/756 valid atoms，weighted RMSD `1.092659`、max group RMSD `1.610383`、7596/7596 pair distances 超差、max pair error `2.803546`。完整证据位于 `/tmp/ofoldx-protenix-parity-compat116/reports/`。
- 验收条件：逐字段确认 dtype、terminal mask 和 component reference geometry 的生成根因；修复或明确契约后，两样本相应 regression 与 strict compare 达到约定结果。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 最终复审确认保持 Investigating；差异真实且可重复，但尚不能定性为 OFoldX production bug。

#### History

- 2026-07-28：状态记录为 `Investigating`；真实差异已稳定复现，但 production 根因尚未确认。
- 2026-07-28：Tik 最终复审确认调查边界，状态保持 `Investigating`；现有证据不足以将差异定性为 OFoldX production bug。
- 2026-07-29：在 current OFoldX commit `16f39a20af01caeeb9189ac5d9800842aa8251c7` 与 baseline commit `c3bfc365b3e1341a11935eddfe7bfdc308092147` 上完成 fresh R2 revalidation。Clean source tar SHA256 为 `d12524a7cef5f66a6f5ea813aa74415a7160ed4fa5749abbf4aacba2779b9140`，wheel SHA256 为 `ea2be0a190952318d10ac6a17f8aceb04cafcdb76aa8bebc60b695342f3176d5`，provenance gate 均为 `PASS`，结果根为 `/tmp/ofoldx-protenix-parity-fresh-4ame-real-msa-20260729-r2`。`ref_element`、`ref_charge`、`ref_atom_name_chars` 仍为 `DTYPE_MISMATCH_ONLY`；`ref_mask` 仍为 `DTYPE_AND_VALUE_MISMATCH`，唯一 mismatch 是 atom index 482、末端 `ALA/OXT`，Protenix `True`、OFoldX `False`；`ref_pos` 仍为同 shape/dtype 下 `VALUE_MISMATCH`，1449/1449 raw values 不同。双 mask 482 atoms、66 spaces 的 weighted RMSD `0.9250821244263586`、max group RMSD `1.5705458658139655`、pair-distance max error `3.573474639007266` 与旧证据一致，分类为 `unchanged`；由于 dtype、mask value 与 geometry 仍不一致，状态保持 `Investigating`。

### BUG-20260728-013: 旧 parity 验证未覆盖真实 MSA 数据路径

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix real-MSA input tensor parity 验证
- 状态：Closed
- 严重级别：High
- 影响：旧报告虽比较了 `msa` 等字段，但 exporter 明确关闭 MSA，得到的 `[1,N]` 只是 query-only fallback，可能被误读为 real-MSA parity 通过。
- 复现步骤：检查旧 manifest 的 `configuration.msa=false`，并观察 `4ame` 与 `1zev` 的 `msa` 第一维均为 1。
- 预期行为：真实 MSA 使用独立 CLI/artifact 路径，显式绑定数据库、来源、entity 和 depth；缺失来源时禁止 query-only silent fallback。
- 实际行为：旧 harness 只有 MSA-disabled exporter/compare，没有区分 base-feature query-only 与 real-MSA 证据。
- 根因：初始 scope 将 MSA 关闭，但报告和 CLI 没有把该限制与真实 MSA 验证路径结构化分开。
- 修复位置：`scripts/parity/protenix_data_parity.py` 新增 real-MSA export/compare 命令、provenance、guards、suffix diagnostic 与 regression tests；consistency report 分开记录 disabled 和 real-MSA 结论。
- 验证证据：双环境 self-test 26/26；`4ame` 双侧各两次 real export 成功且 compare exit 1；`1zev` 四个 export 均写入 `REAL_MSA_BLOCKED`，compare exit 2；旧 `/tmp/ofoldx-protenix-parity-compat116` 证据未被覆盖。
- 验收条件：real-MSA CLI 不复用 disabled artifact 路径；4ame 五字段和 query-row provenance 可审计；1zev 不产生 query-only 假成功；双环境 regression 通过。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 最终复审确认全部验收证据。

#### History

- 2026-07-28：状态记录为 `Open`；Tik 指出旧 23-core 表不能证明 real-MSA parity。
- 2026-07-28：状态由 `Open` 更新为 `Fixed`；独立 CLI、真实 artifacts、self-tests 和 sample-first 报告完成，等待 Tik 关闭。
- 2026-07-28：状态由 `Fixed` 更新为 `Closed`；Tik 最终复审确认全部验收证据。

### BUG-20260728-014: Protenix 与 OFoldX 的 query-row policy 及 RNA real-MSA 覆盖缺口

- 日期：2026-07-28
- 来源任务：OFoldX 与 upstream Protenix real-MSA input tensor parity 验证
- 状态：Investigating
- 严重级别：High
- 影响：`4ame` 的三个 row-axis MSA 字段因 48 对 47 发生 shape `FAIL`；`1zev` 则没有可比较的真实 RNA MSA，当前不能形成跨实现 RNA parity 结论。
- 复现步骤：对同一 47-row `mmseqs_other_hits.a3m` 分别运行 Protenix `MSAFeaturizer` 与 OFoldX `read_msa` + bare MSA；另对 `1zev` 请求 real-MSA export。
- 预期行为：跨实现契约应明确是否在 A3M row 0 之外另发 synthetic paired query，并为 RNA 提供可绑定的同源资产/API；在契约未统一时应据实失败或阻塞。
- 实际行为：Protenix 输出 synthetic paired query + 47-row unpaired pool，得到 `[48,66]` 且 query 重复；OFoldX 输出 47-row pool 一次，得到 `[47,66]`。`1zev` 所需的 `rna_msa/rna_sequence_to_pdb_chains.json` 与 `rna_msa/msas` 均缺失；本次选择的 base `ProtenixProcessor._normalize_msa` 仅支持 supplied protein MSA 并拒绝 RNA。另有支持 protein/RNA 的 `ProtenixTemplateProcessor._normalize_template_msa`，但它是 separate、out-of-scope experiment。
- 根因：4ame 的 shape 差异已确认来自两侧自然 query-row assembly policy；是否应统一 production contract 尚未决定。1zev 是当前数据资产与所选 processor surface 的组合缺口，不代表所有 OFoldX Protenix processor 都不支持 RNA。
- 修复位置：production 修复不在本轮允许范围；harness 只保留自然输出、结构化 provenance 和 diagnostic，不删 A3M row 0、不复制 pool、不后处理张量。
- 验证证据：`profile`/`deletion_mean` exact；Protenix `[1:]` 与 OFoldX `msa`/`has_deletion` exact，`deletion_value` max abs `2.9802322387695312e-08`；双侧 repeat hashes 稳定。1zev report 为 `REAL_MSA_BLOCKED/NOT_COMPARABLE`。
- 验收条件：所有者明确 query-row production contract 并补齐对应 regression；提供 1zev 可验证的 RNA mapping/assets 和 OFoldX API 路径后重新执行双侧真实 parity。
- Owner：Tok
- Tik 复审：2026-07-28 Tik 最终复审确认 harness 与结果；production contract 决策和 RNA 覆盖仍在本轮范围外，当前证据不能将差异定性为任一侧 production bug。

#### History

- 2026-07-28：状态记录为 `Investigating`；4ame 成因已确认，但 production contract 决策与 1zev RNA data/API 能力仍未解决。
- 2026-07-28：Tik 最终复审确认固定提交下的 query-row 根因模式与 harness 结果；production contract 决策和 RNA 覆盖仍在本轮范围外，不能据此定性任一侧 production bug。

### BUG-20260729-015: 沙箱内 Jupyter kernel socket 权限阻断 notebook 执行

- 日期：2026-07-29
- 来源任务：4ame Protenix/OFoldX input tensor notebook 验证
- 状态：Closed
- 严重级别：Medium
- 影响：Tok 在受限 sandbox 内执行 `nbconvert` 时，Jupyter kernel socket 权限错误阻止 kernel 启动，无法生成可信且保留输出的已执行 notebook。
- 复现步骤：在受限 sandbox 内对目标 notebook 运行限定 kernel 与 timeout 的 `python -m nbconvert --to notebook --execute --inplace` 命令。
- 预期行为：限定目标 notebook 的 `nbconvert` 成功执行全部 cell，并在原文件中保留完整输出。
- 实际行为：Jupyter 在 kernel 启动阶段调用 `socket.socket()` 时抛出 `PermissionError: [Errno 1] Operation not permitted`，没有执行 notebook cell。
- 根因：sandbox 的 socket 权限边界阻止 Jupyter 为本地 kernel 分配通信端口，不是项目代码缺陷。
- 修复位置：无源码修复；由 Main 通过权限桥接，仅执行限定目标 notebook 的 `nbconvert` 命令。
- 验证证据：最终 notebook 的 9 个代码单元 execution count 为 1..9、无 error output，nbformat 4.5 合法；SHA256 为 `94734930b2b076f37f5d43a026c1091eaa5fb200c83fbdba18d479967b5fa1ad`。
- 验收条件：不扩大代码改动或权限范围；限定命令成功，输出完整，且 Tik 独立只读复核通过。
- Owner：Tok
- Tik 复审：2026-07-29 Tik 独立只读复核通过，授权 `Closed`。

#### History

- 2026-07-29：状态记录为 `Open`；sandbox 启动 Jupyter kernel 时发生 socket `PermissionError`。
- 2026-07-29：状态由 `Open` 更新为 `Closed`；Main 通过限定命令完成权限桥接执行，完整输出与 Tik 独立验收均通过。
- 2026-07-29：current-code R2 notebook 重跑时再次复现同一 socket `PermissionError`；Main 沿用限定 `nbconvert` 命令完成桥接。最终 notebook 有 13 个代码单元，execution count 为 1..13、全部有输出且无 error output，SHA256 为 `3f3116e8c9a6d767421c8933689e7eb9c1c96c002032fe164c90dc74fce838f3`；既有 workaround 仍有效，状态保持 `Closed`。

### BUG-20260729-016: fresh upstream export 误用不兼容解释器

- 日期：2026-07-29
- 来源任务：4ame current-code Protenix/OFoldX fresh parity revalidation
- 状态：Closed
- 严重级别：High
- 影响：首次 fresh attempt 的两个 upstream exports 在 baseline import 阶段 exit 2，无法产生可信的 Protenix tensors。
- 复现步骤：使用 `/data/users/qinghanw313/miniforge3/envs/ofantibody/bin/python` 和 baseline `PYTHONPATH` 导入 `BaseSingleDataset`。
- 预期行为：export 前的 side-specific gate 应证明解释器满足 baseline 的 pinned data-path 依赖与 API。
- 实际行为：通用 `self-test` 为 26/26，但真实 exporter import 因 Biotite 缺少 `pdbx_convert.PDBX_BOND_TYPE_ID_TO_TYPE` 而失败。
- 根因：实施计划误把 `ofantibody` 解释器当作 upstream pinned environment；harness self-test 不导入真实 exporter 依赖，未暴露该选择错误。
- 修复位置：不修改 production code；R2 改用 `/tmp/ofoldx-parity-env-c52d761-c3bfc365/upstream/bin/python`，并在 export 前验证 `biotite==1.4.0`、`numpy==2.4.1`、`rdkit==2025.9.3`、`torch==2.7.1+cpu`、目标 API、import path 与 `pip check`。
- 验证证据：ATTEMPT1 两份 `failure.json` 作为 provenance 保留；R2 upstream gate V2 为 `PASS`，diagnose 为 `READY`，self-test 26/26，repeat 1/2 export 均 exit 0 且 logical aggregate SHA256 同为 `f976253028c91b6ae81ca860638ecfd7264908bfa41917efebe4a3e8cb591bbf`。
- 验收条件：正确 upstream environment 的 import/API gate、diagnose、self-test 与两个真实 export 全部通过，且不复用失败 attempt 的 tensor artifacts。
- Owner：Tok
- Tik 复审：2026-07-29 Tik 最终只读复审 `APPROVED`，并授权状态 `Closed`。

#### History

- 2026-07-29：状态记录为 `Open`；ATTEMPT1 使用错误解释器导致两个 upstream exports exit 2。
- 2026-07-29：状态由 `Open` 更新为 `Fixed`；R2 使用 pinned upstream environment 完成 gate、diagnose、self-test 与双 repeat export，等待 Tik 授权关闭。
- 2026-07-29：Tik 最终只读复审结果为 `APPROVED`；状态由 `Fixed` 更新为 `Closed`。

### BUG-20260729-017: upstream API gate 错查 parent package namespace

- 日期：2026-07-29
- 来源任务：4ame current-code Protenix/OFoldX fresh parity revalidation
- 状态：Closed
- 严重级别：Medium
- 影响：正确 upstream venv 被错误标记为 gate failure，可能触发不必要的环境重建或错误 blocker 结论。
- 复现步骤：对 `biotite==1.4.0` 断言 `hasattr(biotite.structure.io.pdbx, "PDBX_BOND_TYPE_ID_TO_TYPE")`。
- 预期行为：gate 检查 baseline 实际导入并使用的 `biotite.structure.io.pdbx.convert` namespace。
- 实际行为：目标符号没有由 parent package re-export，package-level 断言失败，但 `pdbx.convert` 上符号存在且 `BaseSingleDataset` 可正常导入。
- 根因：gate 规格没有沿 baseline `parser.py` 的实际 import namespace 定位符号。
- 修复位置：R2 provenance gate 改为 `from biotite.structure.io.pdbx import convert as pdbx_convert` 后检查目标属性；保留 V1 失败日志并分类为 `INVALID_GATE_ASSERTION`。
- 验证证据：`provenance/logs/upstream-gate-v1-invalid-spec.log` 原样保留；`provenance/upstream-gate-v2.json` 记录 incorrect/correct target、`constant_present=true`、`base_single_dataset_import=PASS` 和总状态 `PASS`。
- 验收条件：V1 证据保留，V2 使用实际 namespace 并通过全部版本、API、import path 与 `pip check` 断言。
- Owner：Tok
- Tik 复审：2026-07-29 Tik 最终只读复审 `APPROVED`，并授权状态 `Closed`。

#### History

- 2026-07-29：状态记录为 `Open`；Tik gate 规格错误地检查 parent package。
- 2026-07-29：状态由 `Open` 更新为 `Fixed`；修正 namespace 后 V2 gate 通过，等待 Tik 授权关闭。
- 2026-07-29：Tik 最终只读复审结果为 `APPROVED`；状态由 `Fixed` 更新为 `Closed`。

### BUG-20260729-018: OFoldX raw MSA surface cannot perform auditable multi-entity species pairing

- 日期：2026-07-29
- 来源任务：3x24 Protenix/OFoldX paired-MSA parity 验证
- 状态：Open
- Native scope 更正：3x24 R4 仅为 `VALID_FOR_PROJECTED_LOWERING_ONLY`；因其使用 prepared CIF、query-derived system、外部 pairing plan、component mapping 和 canonical projection，故 `INVALID_FOR_NATIVE_RAW_OR_TRAINING_PARITY`，不得作为 native loss gate 证据。
- 严重级别：Medium
- 影响：无法从两个原始 A3M 文件通过 OFoldX 高层入口完成并审计 species pairing；当前验证只能把外部追踪的配对计划交给 `MSACollection`，不能把结果表述为 raw A3M end-to-end parity。
- 复现步骤：用 `read_msa` 读取两个 protein entity 的 A3M，并尝试从高层 Protenix pipeline 提交多实体 raw-MSA 输入。
- 预期行为：保留每行 header/species 标识，并支持将多个 protein entity 的原始 MSA 交给 pairing stage。
- 实际行为：`read_msa` 只保留投影序列与 deletion counts，丢弃 header/species；高层 raw-MSA surface 对多个 protein entity 报错并要求显式 `MSACollection`。
- 根因：OFoldX 的 raw MSA 数据模型和 reader 没有携带 pairing 所需的 species provenance，多实体 pairing 也没有暴露在高层入口。
- 修复位置：未修改 production code；验证 harness 将 `RAW_A3M_END_TO_END_PAIRING` 标为 `NOT_IMPLEMENTED`，并把 `TRACED_MSA_COLLECTION_LOWERING` 作为独立合同。
- 验证证据：3x24 有两个不同 protein entity、671 个共享 species、1332 个 paired rows；外部追踪计划与 Protenix `MSAPairingEngine` 的 source row/order exact，且 OFoldX `MSACollection` lowering 六字段 MATCH。
- 验收条件：OFoldX reader/data model 保留 header/species provenance，高层接口可直接接受多个 entity 的 raw MSA，并在不注入外部配对计划的条件下复现 Protenix row identity/order 和字段值。
- Owner：Tok
- Tik 复审：等待最终只读复审。

#### History

- 2026-07-29：确认 raw end-to-end blocker；显式 lowering 验证不受阻，状态记录为 `Open`。
- 2026-07-29：native loss 复审发现 R4 输入边界不是 native raw/training；追加 scope 更正，保留 projected lowering 结论但禁止外推为 native pairing/training/loss parity。
- 2026-07-30：runner-only R1 曾记录 `Runner.train()` 没有 per-record MSA loader、两个固定 case 在 raw reader 先失败；旧外部 pairing plan 证据统一降级为 `PROJECTED_DIAGNOSTIC_ONLY`。
- 2026-07-30：Tik current-code provenance 复审判定 runner R1 为 `STALE_WHEEL_INVALID`；上述 R1 ingestion 观察正式撤回，不能作为当前源码证据，状态保持 `Open`。

### BUG-20260729-019: paired-MSA artifacts 未强制绑定同一 harness 版本

- 日期：2026-07-29
- 来源任务：3x24 Protenix/OFoldX paired-MSA parity 验证
- 状态：Closed
- 严重级别：High
- 影响：旧结果目录中的四次导出没有在 manifest 中统一记录并强制校验 harness 绝对路径与 SHA256，无法证明 compare 使用的代码就是所有 artifacts 的生成代码。
- 复现步骤：检查旧目录 `/tmp/ofoldx-protenix-paired-msa-3x24-20260729` 的四份 manifest，并尝试在缺少或漂移 harness provenance 时执行 compare。
- 预期行为：四份 manifest 均记录同一绝对 harness 路径与 SHA256；任一缺失、路径不同或哈希漂移都使总体比较失败。
- 实际行为：旧 manifest 未建立该强制合同，旧目录不能作为最终证据。
- 根因：artifact provenance 记录了环境、commit 和输入文件，但遗漏了直接生成 artifacts 的 harness 自身版本。
- 修复位置：`scripts/parity/protenix_data_parity.py` 的 artifact writer 与 paired report aggregate；旧目录添加 `SUPERSEDED_FAILED_PROVENANCE_ATTEMPT` 标记，结果从全新固定目录重跑。
- 验证证据：双环境 37/37 self-tests；`r2` 四份 manifest 的 harness path 均为 `/data/users/qinghanw313/AlloDesign/scripts/parity/protenix_data_parity.py`，SHA256 均为 `137901ad2d3381b6e3fa6a68c24348ca0c91b81554107c474acfc5b5abb54346`；report `harness_provenance.classification=MATCH`。
- 验收条件：缺失/漂移 provenance 的 focused tests 失败；四次 fresh export 与 compare 通过；最终 notebook 和报告只引用 `r2`。
- Owner：Tok
- Tik 复审：2026-07-29 Tik 最终只读复审 `APPROVED`；四份 manifest 的 harness path/SHA 一致，缺失或漂移会使比较失败，授权关闭。

#### History

- 2026-07-29：Tik 复审发现旧 artifacts 无法证明统一 harness 版本，状态记录为 `Open`。
- 2026-07-29：加入强制 manifest/aggregate 检查并完成全新 `r2` 四次导出，状态由 `Open` 更新为 `Fixed`。
- 2026-07-29：Tik 最终只读复审结果为 `APPROVED`；状态由 `Fixed` 更新为 `Closed`。

### BUG-20260729-020: paired-MSA 报告未覆盖完整 model input dict

- 日期：2026-07-29
- 来源任务：3x24 Protenix/OFoldX paired-MSA 全字段 parity 验证
- 状态：Closed
- 严重级别：High
- 影响：旧 paired-MSA 报告只展示 MSA lowering 字段，无法回答两个 model input dict 的所有 key、raw shape/dtype、投影后数值差异，也会把 raw dict 整体不等与可比较字段结论混在一起。
- 复现步骤：检查旧 `3x24-paired-msa.json` 的 `field_comparisons`；其中只有 MSA、deletion 和 pairing trace，未注册 Protenix 43 个与 OFoldX 30 个原始字段。
- 预期行为：独立注册全部原始字段，在动态 canonical token/atom identity 上比较可对应字段，显式报告 derivation、无 counterpart、semantic boundary、dtype/value 差异，并将 raw dict 整体 verdict 与 traced MSA lowering verdict 分开。
- 实际行为：原 paired-MSA builder 只实现专用六字段 lowering 合同；通用 parity builder 又绑定固定样本轴和另一套字段映射，不能直接用于 3x24 modified-residue/multimer 轴。
- 根因：paired-MSA 验证最初以 MSA 模块为边界，没有建立全 raw field registry、动态 identity projection 和独立完整性 gate。
- 修复位置：R3 因 raw value consumption 不完整已作废；R4 在 `scripts/parity/protenix_data_parity.py` 加入真实 raw `token_index` 读取、`reference_mask`/packed duplicate gate、`token_features[:,0:384]` row，以及逐 root selector/gate consumption audit；notebook 与 Markdown 只读取 R4。
- 验证证据：双环境 self-test 各 47/47；fresh R4 两侧双重复 exact；report integrity、全部 fixed gates 与 raw value consumption 均 `PASS`，raw coverage 43/30，公共轴 419 tokens/3275 atoms，56 行分类分布精确为 MATCH 22、dtype-only 3、dtype+value 1、value 3、no-counterpart 25、semantic 2。notebook 外部 parser 验证 P43/O30 全字段、37,030,861 个值、全部 logical SHA、零 ellipsis/零 error。当前 paired-MSA notebook SHA256 为 `abc7da9dcafbf53ce66137c3d88acc45839d7265740f19d9c57f5c0e029b953c`，Markdown SHA256 为 `144aa83df8f4d94e2a1612fb8e726e4ca657f457d7ccdfea697fa1d921e52241`。
- 验收条件：未知或缺失 raw field 必须失败；报告固定为 56 行且覆盖全部 43/30 root fields及其实际 value selector；复合字段 gap/overlap/out-of-range/declared-but-unread 必须失败；raw dict、traced lowering 与 raw-A3M boundary 三个 verdict 独立；notebook 完整打印并可重放验证每个字段。
- Owner：Tok
- Tik 复审：2026-07-29 Tik R4 最终独立只读复审 `APPROVED`，授权关闭。

#### History

- 2026-07-29：确认 paired-MSA 报告只覆盖 MSA 子集，状态记录为 `Open`。
- 2026-07-29：完成独立全字段实现、fresh R3 双重复、55 行报告和完整 notebook dump，状态由 `Open` 更新为 `Fixed`，等待 Tik 授权关闭。
- 2026-07-29：Tik 最终复审判定 `REVISE`；R3 只证明 root 名称覆盖，未证明所有 raw value 被实际消费，状态由 `Fixed` 退回 `Open`，R3 标记为不完整证据。
- 2026-07-29：完成 R4 真实 value consumption 实现、fresh 双重复、56 行报告和完整 notebook dump，状态由 `Open` 更新为 `Fixed`，等待 Tik 最终只读复审授权关闭。
- 2026-07-29：Tik R4 最终独立只读复审结果为 `APPROVED`；状态由 `Fixed` 更新为 `Closed`。
- 2026-07-29：追记当前交付文件哈希；paired-MSA notebook=`abc7da9d...b953c`，Markdown=`144aa83d...2241`，不改变已批准的 R4 结论。
- 2026-07-30：scope 纠正不否定 R4 对“外部计划+canonical projection 后全字段”的覆盖，但撤回其 native MSA/pairing 含义；renderer 与 notebook 不再输出裸 native `MATCH/PASS`。
- 2026-07-30：4ame generator 加入幂等 adapted/projected-only transform；旧 R2 JSON 本体同步写入 native alignment `NOT_TESTABLE`、pairing `INVALID/SUPERSEDED`，不再保留 `CONFIRMED/PASS_EXACT` 正向标签。

### BUG-20260729-021: OFoldX native pipeline blocks all three audited loss-parity cases

- 日期：2026-07-29
- 来源任务：Protenix/OFoldX same-output native loss parity 验证
- 状态：Open
- 严重级别：High
- 影响：`3x24`、`4ame`、`1zev` 均无法完成 OFoldX native raw-to-training pipeline，因此不能建立完整 token/atom bijection、target contract、相同 output-head shape，也不能执行 `<1e-7` same-output native loss 比较。
- 复现步骤：将原始 `/data/shared/datasets/protenix/mmcif/{3x24,4ame,1zev}.cif` 逐一传给无额外参数的 OFoldX `read_structure(raw)`，成功时继续 production MSA binding、`ProtenixProcessor.process_training` 与 `collate_denoising`；每个 case 独立重复两次。
- 预期行为：raw reader 成功后继续 native MSA binding、`ProtenixProcessor.process_training` 和 `collate_denoising`。
- 实际行为：`3x24` 在 reader 因 `molecule_1` molecular definition 冲突失败；`4ame` 在 reader 因 `A:23:NH2` formal charge 冲突失败；`1zev` reader 成功，但 training processor 因 `HOH` 没有 explicit reference coordinates 失败。每个 case 两次异常和终止阶段完全一致。
- 根因：前两例是 OFoldX raw structure reader 与原始 mmCIF 语义冲突；第三例是 OFoldX Protenix processor 不隐式生成 `HOH` reference coordinates。当前任务禁止通过 component injection、mapping、过滤或结构重建改变 native input 边界。
- 修复位置：未修改 production code；`scripts/parity/protenix_native_pipeline_parity.py` 和 `scripts/parity/protenix_loss_parity.py` 以 fail-closed gate 保存证据并返回 `NOT_TESTABLE`。
- 验证证据：fresh R6 pipeline report SHA256 `b9ab8ae25042a01d320f9a76a5c6700ab536be17055c5015a8c1f484c5b1430c` 与 R7 loss report SHA256 `1cfaee034d44d3c7812e16e6eeb5b256bb2b927b2d53cda43794e18c4b1a12a4`；12 份双侧 status、六组 repeat determinism MATCH、独立 production-bin probes、递归 status/harness/source/input/config/index/A3M/payload/prediction hash gate 和 focused tests 21/21 均通过。R6/R7 取代 R5/R6 evidence roots。
- 验收条件：至少一个 pairing-capable case 在不注入或转换输入的条件下完成双侧 native raw/training pipeline，并通过完整 token/atom bijection、target coordinate/mask/index exact 和 output-head shape exact gate。
- Owner：Tok
- Tik 复审：等待复审。

#### 2026-07-30 current-code direct reader 补充证据

- 环境与输入边界：CPU-only；直接读取 raw `/data/shared/datasets/protenix/mmcif/3x24.cif`；未使用 parity adapter。CIF SHA256 为 `e7e7636c3669ebf9cf4158fdd9bc9457efa8aeea2855f63acde73f8167522374`，源文件未修改。
- 最小复现：

```python
from ofoldx.io import read_structure

path = "/data/shared/datasets/protenix/mmcif/3x24.cif"

read_structure(path)
read_structure(path, atom_layout="observed")
read_structure(path, atom_layout="complete_heavy")
read_structure(path, atom_layout="complete_all")
```

- 预期行为：四种读取模式都应成功解析同一份 raw mmCIF，或对文件中缺失信息作明确且可处理的模式级响应。
- 实际行为：默认模式与 `observed` 均抛出 `ValueError: mmCIF covalent bond order is missing`；`complete_heavy` 与 `complete_all` 均抛出 `KeyError: 'FE'`。
- 影响：`3x24` 无法通过直接 production reader 进入后续 native structure、MSA、training 或 loss 流程；不同模式在 reader 内以两类异常终止。
- 根因边界：本次仅确认按模式稳定复现的异常边界，未验证根因，不对缺失 bond order 或 `FE` 查找失败的内部来源作推断。
- 复现证据：`notebooks/io.ipynb`；该 notebook 使用上述 raw input 和直接 `ofoldx.io.read_structure` 调用。

#### History

- 2026-07-29：状态记录为 `Open`；初版 R1 错误地将三例都归因为 reader/component 问题。
- 2026-07-29：R2 使用 production 默认 `read_structure(raw)` 重跑；更正为 `3x24`/`4ame` reader 失败，`1zev` reader PASS 后在 processor 失败。
- 2026-07-29：R3 按官方两段 config merge（最终 `N_cycle=10`）全新双重复重跑，所有失败边界与 R2 一致，且完成全量递归 provenance/hash 校验；R4 loss audit 继续保持 `NOT_TESTABLE`。
- 2026-07-29：R4/R5 加入 `protenix_template` registry、双侧 ref augmentation=false、head dtype/bin boundary、target dtype/value、actual index/A3M path hash 与 OF-only native E2E gate；失败边界不变，状态保持 `Open`。
- 2026-07-29：R5/R6 加入 exact production bin 公式、`atom_index` injection 语义、status child 递归 verifier 和落盘后新进程 E2E consumer；三例失败边界仍不变，状态保持 `Open`。
- 2026-07-29：R6/R7 用两个 pinned runtime 分别直接调用 production decoder，并直接调用四个 Protenix label method；三例 native pipeline 失败边界仍不变，状态保持 `Open`。
- 2026-07-30：runner-only R1 曾以 `Runner.train()` pre-hook 运行固定 3x24/4ame，但随后因使用 stale wheel 被判 `STALE_WHEEL_INVALID`；其 `DATASET_RAW_READ` 结果已撤回，不能作为当前源码证据，状态保持 `Open`。
- 2026-07-30：在 CPU-only、raw input、无 parity adapter 的 current-code direct reader 复现中，`3x24` 默认/`observed` 抛出缺失 covalent bond order 的 `ValueError`，`complete_heavy`/`complete_all` 抛出 `KeyError: 'FE'`；记录为新的 reader 边界证据，未推断根因，状态保持 `Open`。

### BUG-20260729-022: 旧 loss R1 使用 synthetic/common contract 产生无效 native parity 数值

- 日期：2026-07-29
- 来源任务：Protenix/OFoldX same-output native loss parity 验证
- 状态：Closed
- 严重级别：Critical
- 影响：旧 `/tmp/ofoldx-protenix-loss-parity-20260729-r1` 的数值来自 synthetic/common contract，而不是两侧 production raw-to-training pipeline；将其引用为 native loss parity 会形成错误的 `<1e-7` 结论。
- 复现步骤：检查旧 R1 输入构造和 loss payload，确认其绕过 native data pipeline 并把共同构造的字段送给两侧公式。
- 预期行为：只有 dual native success、完整 identity bijection、target exact、head shapes identical 与 same prediction hash 全部通过后，才分别调用两个 pinned runtime 的原生 loss consumer。
- 实际行为：旧 R1 在上述 gates 之前计算数值，且 common contract 隐藏了两侧 objective membership 与缺失字段差异。
- 根因：把公式级 diagnostic 当成 production contract parity，缺少 native provenance 与 fail-closed admission gate。
- 修复位置：`scripts/parity/protenix_native_pipeline_parity.py` 与 `scripts/parity/protenix_loss_parity.py`；旧 R1 写入 `STATUS.json`，标记 `SUPERSEDED` 和 `native_loss_parity_evidence=INVALID`。
- 验证证据：fresh native pipeline R6 无 dual success；fresh loss R7 未执行 eligible-case consumer、未评估 threshold，并明确 `NOT_TESTABLE`。旧 roots 全部写入 SUPERSEDED/INVALID；E2E producer 先保存 NPZ/manifest，随后新 OFoldX 进程 formal consumer 重载同一 SHA 并调用 registered criterion；NPZ/manifest tamper 均被拒绝；focused tests 21/21 PASS。
- 验收条件：旧 R1 永不作为 native parity 证据；未来 eligible case 必须通过 R6 类 native gate 并由独立 OFoldX/Protenix consumer 绑定同一 prediction hash。
- Owner：Tok
- Tik 复审：2026-07-29 Tik 第六轮最终简审 `APPROVED`，明确授权关闭。

#### History

- 2026-07-29：发现旧 R1 synthetic/common 输入边界错误，状态记录为 `Open`。
- 2026-07-29：旧根目录已写 SUPERSEDED/INVALID 状态，fresh R2/R3 采用 fail-closed native gate，状态由 `Open` 更新为 `Fixed`，等待 Tik 复审。
- 2026-07-29：fresh R3/R4 以官方 config merge、production consumer、semantic head、atom-index 和递归 hash gate 替代 R2/R3；状态保持 `Fixed`，未越权关闭。
- 2026-07-29：fresh R4/R5 修正 OFoldX template registry 构造、禁止 `coordinate_mini=coordinate` 补齐，并加入 dtype/bin/A3M/status/E2E 负向门；状态保持 `Fixed`，等待 Tik 最终复审。
- 2026-07-29：fresh R5/R6 将 E2E criterion 移到落盘后的新进程 consumer，加入真实 consumed SHA、transitive status-child verifier、exact production bin 和 injection/bijection 门禁；状态仍为 `Fixed`，Tik 批准前不关闭。
- 2026-07-29：fresh R6/R7 以两个独立 pinned subprocess 的 production decoder 直接输出和四个真实 Protenix label method 输出替代共享公式观测；状态仍为 `Fixed`。
- 2026-07-29：记录第五轮 Tik `REVISE`：共享重建不构成双侧 production-call 证据；R6/R7 已完成直接调用、source/output drift gate 和无共享 helper 断言。第六轮 notebook 依赖外部 `display` 的问题也已修复，状态保持 `Fixed`，等待第六轮简审。
- 2026-07-29：Tik 第六轮最终简审 `APPROVED` 并明确授权；状态由 `Fixed` 更新为 `Closed`。

### BUG-20260729-023: formal harness 启动目录与 notebook socket 限制

- 日期：2026-07-29
- 来源任务：Protenix/OFoldX same-output native loss parity 第四轮验证
- 状态：Fixed
- 严重级别：Low
- 影响：第一次 R4 Protenix 启动从仓库根运行且未设置 baseline `PYTHONPATH`，在任何 sample status 生成前报 `ModuleNotFoundError: scripts`；Jupyter `nbconvert` 又因 sandbox 禁止创建本地 socket 而无法启动 kernel。
- 复现步骤：直接从仓库根调用 upstream Python 执行 native harness；或在当前 sandbox 运行 `python -m nbconvert --execute`。
- 预期行为：Protenix 入口能导入 baseline `scripts.prepare_training_data`；notebook 的六个代码块能按顺序验证。
- 实际行为：前者在 import stage 退出；后者在 `jupyter_client.find_available_port()` 报 `PermissionError: Operation not permitted`。
- 根因：Protenix upstream 的源码布局依赖 baseline cwd/PYTHONPATH；notebook kernel transport 依赖本地 socket，而 sandbox 禁止该系统调用。
- 修复位置：正式命令固定使用 cwd `baseline/Protenix` 和显式 baseline `PYTHONPATH`；notebook 使用同一 Python namespace 普通顺序执行六个代码块。
- 验证证据：首次错误启动仅创建两个会被确定性重建的 runtime symlink，未创建 sample/status/report；正确启动后 R4 12 份 status 完整且所有哈希 gate PASS。notebook 普通顺序执行 `6/6 PASS`，报告 SHA 与正式 artifact 一致。
- 验收条件：正式 R4/R5 artifact 无不完整 status 混入；notebook 所有代码块在无 kernel socket 的验证方式下通过。
- Owner：Tok
- Tik 复审：等待复审。

#### History

- 2026-07-29：问题出现并在同轮修正，状态记录为 `Fixed`；等待 Tik 复审。

### BUG-20260729-024: E2E persisted sigma 校验首次补丁插入错误函数

- 日期：2026-07-29
- 来源任务：Protenix/OFoldX same-output native loss parity 第五轮验证
- 状态：Closed
- 严重级别：Low
- 影响：第一次 unit run 中，`persisted_sigma_data` 校验误插入 `smoke_ofoldx`，而 `consume_ofoldx` 只引用未赋值变量；导致 smoke 失败且新进程 consumer 在写结果时 `NameError`。
- 复现步骤：运行第五轮初版 `NativeLossParityUnitTest`；观察 16 项中 1 FAIL、1 ERROR。
- 预期行为：只有 formal consumer 从重载 NPZ 读取 `sigma_data`，并与 registered config 对照；smoke 不依赖持久化 payload。
- 实际行为：补丁按宽泛上下文命中第一个同形 registry/config 代码块。
- 根因：两个函数具有相同 registration/config 前缀，manual patch 上下文不够唯一。
- 修复位置：从 `smoke_ofoldx` 删除该逻辑，并在 `consume_ofoldx` 的 NPZ load 与 registered config 构造后赋值和校验。
- 验证证据：修复后 unit 16/16 PASS；E2E 单测 1/1 PASS；formal consumer 状态 `PASS_PRODUCTION_REGISTERED_CRITERION_AFTER_DISK_RELOAD`，loss `1.6037206649780273`，实际 consumed SHA 与 producer NPZ SHA exact；最终 focused tests 21/21 PASS。
- 验收条件：完整 focused tests 21/21 PASS，smoke 与 formal E2E 独立通过，producer 不执行 criterion。
- Owner：Tok
- Tik 复审：2026-07-29 Tik 第六轮最终简审 `APPROVED`，明确授权关闭。

#### History

- 2026-07-29：首轮 unit 暴露 patch 定位错误，立即记录并修复；状态设为 `Fixed`，等待 Tik 复审。
- 2026-07-29：R7 保留修复并由 persisted E2E、tamper tests 与 21/21 focused tests 复验，状态保持 `Fixed`。
- 2026-07-29：第五轮 `REVISE` 来自独立 production-call 证据不足，而非本 bug 回归；R6/R7 修复该 blocker。第六轮补齐 notebook 内部 `display` 导入和 clean-globals 6/6 execution record，本 bug 保持 `Fixed`。
- 2026-07-29：Tik 第六轮最终简审 `APPROVED` 并明确授权；状态由 `Fixed` 更新为 `Closed`。

### BUG-20260730-025: 旧 adapted/projected MSA 证据误导且 current-code native loss 被打包缺陷阻断

- 日期：2026-07-30
- 来源任务：Protenix/OFoldX runner-boundary input 与 loss parity 验证
- 状态：Open
- 严重级别：Critical
- 影响：旧 4ame R2 和 3x24 R4 把 adapted structure、单实体 A3M 策略、外部 pairing plan 与 canonical projection 的结果显示成裸 `MATCH/PASS`，可能被误读为 native runner MSA/input/loss parity；current-code fresh wheel 无法导入，因此不能验证 Runner 的实际 ingestion 或 loss parity。
- 复现步骤：固定原始 3x24/4ame CIF，分别在两个独立进程构造 `Runner(config)`，在实际 `runner.model` 注册 pass-through `forward_pre_hook(with_kwargs=True)`，调用 `runner.train()`；不传 prepared CIF、MSA object/path、外部 pairing 或投影。
- 预期行为：两侧完成 native one-epoch，hook 捕获可关联的模型输入；Runner 从公开 data config 加载每条记录的原生 MSA。
- 实际行为：R1 的 raw-reader/ingestion 结果因 stale wheel 已撤回。current-code R2 四个 OFoldX worker 均在导入 `ofoldx` 时因缺少 `ofoldx._version` 失败，hook=0；fail-closed 规则下 Protenix 未启动，输入与 loss 均未生成。
- 根因：当前源码构建的 wheel 未包含运行时必需模块 `ofoldx._version`。旧 harness 还曾绕开 native 边界且没有充分隔离 diagnostic scope，但其 R1 ingestion 结论不再有效。
- 修复位置：不修改 production；新增 `scripts/parity/protenix_runner_input_parity.py`，旧 renderer/report/notebook 加 scope tombstone，4ame native aligned/root-cause 外推已撤回。
- 验证证据：R1 已标 `STALE_WHEEL_INVALID`。current-code R2 动态绑定 clean HEAD、tracked tree、wheel/RECORD、extracted snapshot、installed distribution 和 `direct_url`，source closure 为 `PASS`；fresh wheel `744017e6...b6b807` 因缺少 `ofoldx._version` 无法导入，四个 OF worker 均在 `CURRENT_CODE_RUNTIME_IMPORT` fail closed，Protenix 未启动，正式结论仍为 `NOT_TESTABLE`。
- 验收条件：至少一个固定 case 在无适配、注入、投影条件下完成双侧 native runner epoch；OFoldX Runner 公开配置能够加载 per-record MSA 并保留 pairing provenance；两侧 hook 捕获通过 identity/target/output gate 后才允许数值 loss 比较。
- Owner：Tok
- Tik 复审：等待复审。

#### History

- 2026-07-30：问题在 runner-only 复审中确认，状态记录为 `Open`；按用户要求遇到 raw/ingestion bug 后停止，不扩展 case。
- 2026-07-30：Tik 指出 R1 使用旧 compat wheel，R1 改标 `STALE_WHEEL_INVALID`。R2 从当前 clean tracked OFoldX 源码构建并安装 fresh wheel；新 blocker 是 wheel 未包含被 `.gitignore` 排除的 `ofoldx/_version.py`，状态保持 `Open`。
- 2026-07-30：Tik 发现 `compare-real-msa` 成功写盘仍可能使用 scope 前的局部 `count_checks` 和旧 `summary_lines`。writer 现对最终 report 递归执行幂等 scope，并只用 `_render_adapted_real_msa_summary(report)` 生成 TXT；新增真实 CLI/writer-branch 回归后 data self-test 51/51。harness SHA256=`eacb21d6885d14a88956cc15ffcde1ba10bc67f2b60ee2396c0245c13f468baa`；现有 4ame JSON/TXT 幂等重写后 SHA256 仍分别为 `43382cc059a6a4d9e5c0e773625d8c6664ae8b986344d7840b1a993e2c9d25bb`、`abf9270287342012e26e3bd7efbd32d37fb3103328921f44e4d3be7ce9285338`，JSON/TXT 裸 `PASS`/`PASS_EXACT`/`PASS_TOLERANCE`/`CONFIRMED` token 均为 0。

## Bug Template

<!--
### BUG-YYYYMMDD-NNN: 简短标题

- 日期：
- 来源任务：
- 状态：
- 严重级别：
- 影响：
- 复现步骤：
- 预期行为：
- 实际行为：
- 根因：
- 修复位置：
- 验证证据：
- 验收条件：
- Owner：Tok
- Tik 复审：

#### History

- YYYY-MM-DD：状态由 `<old>` 更新为 `<new>`，原因：
-->
