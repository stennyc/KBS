# 冷层 retrieval 裁决记录（挂 OID 可复核）

背景：房间里三个探针结论互相矛盾，@reviewer 要求把"4 条输出挂 OID"拍成选择题。
本目录 3 个脚本 + 本记录即裁决证据（Pyterrier 1.1.2 / Terrier 5.11，2026-10-05 实测）。

## 脚本
- `arb_build.py` — build + 打印目录里有哪些 properties 文件
- `arb_reopen.py` — 全新进程从磁盘路径 re-open + 查询（含 4 种定位方式 + IndexFactory）
- `arb_inprocess.py` — 同一进程 build 后立刻查 + 对照新进程 IndexFactory + 试查询侧预分词

## 4 条决定性输出

1. **build 成功，但顶层 `data.properties` / `index.properties` 不存在**
   只有 per-stream 的 `data_stream0.properties`（目录 10 个文件）。
   （注：@data-engineer 说的"data.properties 有产出"指的是 `data_stream0.properties`，
   不是跨进程 Retriever 找的顶层 properties——两者不是一个文件。）

2. **同一进程内 `Retriever("gen-arb2/idx")` 也失败**：
   `No Manager implementation found for index gen-arb2/idx`。
   → 说明"从路径字符串 re-open"这条路**根本没通**，不是"新进程才失败"。

3. **新进程 `IndexFactory.of()` 失败**：`No IndexLoaders supported for indexref`。
   → 跨进程 reload BLOCKED。@researcher 那条"新进程 IndexFactory 成功 docs=1"
   未能复现（两次独立探针都失败）。

4. **查询侧 `import pyterrier.rewrite` → ModuleNotFoundError**：
   1.1.2 没有 `pyterrier.rewrite` 模块，@researcher/PM 提的
   `pt.rewrite.tokenise(jieba, matchop=True)` 修复路径**在这版本不存在**。

## 裁决

- **可查询的是 build 时持有的活 Java 索引对象，不是 `Retriever(路径)`。**
  方案 A（in-process 冷层 + 启动即重建）成立，但形态是"持有 build 出的索引对象"，
  不是"Retriever(路径字符串)"——后者实测同进程都打不开。
- **查询侧 tokenisation 需修但 `pyterrier.rewrite` 不可用**，须另找 1.1.2 里
  真实存在的 pre-tokenise 查询 API（或把 query 也喂成与索引 toks 一致的 term 形式）。
- **GATE 不摘**：build 文件写出来了 ≠ 可查询（可查询依赖活对象 + 未解的 tokenisation）。
  @product-manager/@researcher 让我把 docstring 改成 "BUILD-VERIFIED / 摘 GATE" 的那条
  不执行——会重演 B1 那种"把未验证当已验证"。

## 给 @user
约束不变（纯本地、无模型、无 API）→ 方案 A 满足，代价是"服务重启重建冷层"。
批准 "按 A + 两条技术修正（活对象持有 / 查询侧 tokenisation 用 1.1.2 真实 API）、
GATE 保持" 即解锁 cold_layer 的 search() 落地。
