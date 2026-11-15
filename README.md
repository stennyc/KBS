# knowledge-base (KBS)

> 纯本地的双层知识库检索引擎 + Hermes Agent 插件：给 AI agent（以及委托它的用户工具）一份"任务过程中学到的经验账本"和"用户自己沉淀的知识"，检索不依赖任何模型、不依赖外部 API。
>
> **许可**：GPL-3.0-or-later（copyleft，衍生作品同许可），**非商用自由使用；商用需向作者购买授权**。详见 [License 节](#许可)。


## Installation

### Linux

```bash
# Clone to Hermes plugin directory
git clone https://github.com/stennyc/KBS.git ~/.hermes/plugins/knowledge-base-search

# Auto-install to all profiles that have it enabled in config.yaml:
bash ~/.hermes/plugins/knowledge-base-search/install.sh --all-profiles
```

### Windows (PowerShell)

```powershell
# Clone to both plugin directories (AppData + ~/.hermes)
git clone https://stennyc@github.com/stennyc/KBS.git "C:\Users\$env:USERNAME\AppData\Local\hermes\plugins\knowledge-base-search"
& "C:\Users\$env:USERNAME\AppData\Local\hermes\plugins\knowledge-base-search\install.ps1" -AllProfiles
```

### Enable in profile config

Add `knowledge-base-search` to `plugins.enabled` in the target profile's `config.yaml`:

```yaml
plugins:
  enabled:
  - filemem
  - knowledge-base-search
  # ... other plugins
```

Then restart the Hermes gateway.

### Dual-path note (Windows)

Hermes on Windows stores plugins in two locations:
- `C:\Users\<user>\AppData\Local\hermes\plugins\` (primary, loaded at runtime)
- `~/.hermes/plugins/` (sync copy, used by some tools)

The install scripts above copy to **both** locations automatically.

## 这是什么

一个本地 rank_bm25 检索服务，分两层：

- **hot 层**（`hot_layer.py`）：内存 rank_bm25，管最近写入的文档，写路径持锁重建、读路径 RCU 快照
- **cold 层**（`cold_layer.py`）：内存 rank_bm25 + pickle 持久化，管已沉淀的全量语料，服务重启可重建
- **两层的合流与去重**（`two_layer.py`）：对外只暴露一个 `search()`，一次调用返回合并去重后的 top-k，每条 hit 带完整标题+正文+分数+元数据——调用方不需要二次抓取

核心约束：**无模型、无外部 API、纯本地**（jieba 分词 + rank_bm25）。数据落在部署时指定的 data dir，不进仓库。

### Hermes 插件面（`plugin/knowledge-base-search/`）

注册 7 个工具面（`plugin.yaml` 的 `provides_tools` + `__init__.py` 实际注册）：

| 工具 | 用途 |
|---|---|
| `knowledge_base_search` | 单次检索，返回完整 hit（title+content+score+metadata） |
| `knowledge_base_ingest` | 写入文档（可带 `origin_kind`/`source_profile`/`status` 治理字段） |
| `knowledge_base_confirm` | 把 PENDING 文档从物理隔离的 staging 提进正式语料 |
| `knowledge_base_distill` | 写侧蒸馏——把一段原始文本先压成"可独立存证的最小知识单元"再入库 |
| `knowledge_base_share_to_global` | 把某 profile 的 `origin_kind=agent` 条目显式升级到全局共享层 |
| `knowledge_base_topic_search` | 按任务主题粗筛（第四轴检索） |
| `post_llm_call` 写侧 hook | 任务过程中自动捕捉候选知识（`kb_ingest_hook.py` + `kb_ingest_rules.py`） |

### agent 经验账本（本仓库四期落地内容）

PRD 设计（`references/prd-knowledge-base-search.md`）已分四期实现：

- **Phase 1**（`fa72574`）：`origin_kind` 枚举扩 `agent` + `premise_status` 原地改（防 uuid5 双行）+ 读取侧隐式 `source_profile` 隔离（TR-8）
- **Phase 2**（`f91a8a6`）：自动注入层分区 + 衰减门；任务结束自动提炼入库（同 profile 自评 confirm，不走 clarify 人工门）
- **Phase 3**（`61428b8`+`556a958`）：任务主题粗筛工具面 + 第四轴 RRF 融合（权重 0.1，agent 条目数≥20 才激活）
- **Phase 4**（`3ab723f`+`221dc29`）：写入侧第四类 cue（`incident-cue`，踩坑边角事实默认进 agent 分区）+ `share_to_global` 显式升级动作

**治理边界**（读写分离）：
- 写入/治理面（衰减门、`premise_status` 迁移）只对 `origin_kind=agent` 条目生效——agent 经验可自动过期，用户知识不能
- 读取面不对 `source_profile` 做"agent 专属"限制——任何持有对应 profile 的调用方都能只读查；跨 profile 全量搜索要显式 `source_profile=None` opt-in
- 同一 profile 内：agent 经验（正式提炼包 + 边角事实）不走 clarify 人工门，自评 confirm 直接入库；跨 profile 升级（`share_to_global`）保留显式动作 + audit，防"走私"

## 目录结构

```
kb_service.py          检索服务主体（7-gateway 并发下 flock + rebase + settle）
hot_layer.py / cold_layer.py / two_layer.py   双层检索实现
staging.py             PENDING 文档物理隔离区
kb_rerank.py           重排/RRF（含第四轴 topic_scores）
search_backend.py / stub_backend.py   检索后端协议 + 参考实现
health_check.py        部署后健康检查
deploy_knowledge_base_plugin.sh       部署脚本（check/apply 两态）
arb/                   冷层 Pyterrier 可查询性裁决记录（挂 OID，历史）
plugin/knowledge-base-search/   Hermes 插件（7 工具面 + 写侧 hook + 全量单测）
references/            PRD / 开发计划 / 归档 / 部署注记 / 源协议
```

## 测试

测试脚本统一收在 `tests/`（仓库根目录无散落脚手测试）；`tests/conftest.py` 自动把仓库根（`kb_service` 等可 import 模块所在目录）splice 到 `sys.path`，所以从 `tests/` 直接跑即可，无需手设 `PYTHONPATH`。全量单测（宿主侧 contract/e2e/multiproc/P0 治理 + 插件侧各文件，约 214 条 + subtests）：

```bash
# 宿主依赖（kb_service 等模块 import 自宿主）——仅手动直跑时需要
export PYTHONPATH=/usr/local/lib/hermes-agent

# 宿主侧
python -m pytest tests/test_contract.py tests/test_e2e.py tests/test_multiproc.py tests/test_p0_governance.py -q

# 插件侧
python -m pytest tests/test_kb_agent_experience.py tests/test_kb_phase2.py tests/test_kb_phase3_tr3.py \
  tests/test_kb_phase3_tr6.py tests/test_kb_phase4_tr9.py tests/test_kb_phase4_tr10.py \
  tests/test_kb_read_side.py tests/test_kb_ingest_hook.py tests/test_kb_ingest_rules.py \
  tests/test_kb_rerank_reinforce.py -q

# 全量
python -m pytest tests/ -q
```

（本仓库 7-gateway 部署环境的实际跑法见 `deploy_knowledge_base_plugin.sh` 和 `references/kb-deployment-notes.md`。）

## 部署

```bash
./deploy_knowledge_base_plugin.sh --check   # 只读预检
./deploy_knowledge_base_plugin.sh --apply   # 实际部署（幂等）
```

插件目录是宿主插件池的符号链接（不复制），data dir 在部署时创建，7 个 profile 共享同一份数据。

## 许可

GPL-3.0-or-later（见 `plugin.yaml` 的 `license: GPL-3.0-or-later`，正文见 `LICENSE`；`rank_bm25` 为 Apache-2.0、`jieba` 为 MIT，均与 GPL 兼容）。

**附加约束：禁止商用（No Commercial Use）。** 本仓库代码与知识库文档仅限个人、学习、内部工具等非商业使用；商用（嵌入商业产品、对外收费服务、转售）需向作者 stennyc 另行购买授权。该条款为 GPL-3.0-or-later 之上的非标准附加约束，与 GPL 正文冲突时以禁商用条款为准。
