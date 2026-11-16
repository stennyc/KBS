"""kb 侧内联共享同步 — 只推不拉版 (owner 定稿 2026-10-22 三扇门重构, 与
filemem shared_memory.py 对称, X 方案两仓各一份).

kb 与 filemem 对称的只推不拉同步. 中央库 = 同一台 Postgres 的 kb_docs 表
(全量沉淀, append 语义, 不做跨设备门控覆盖).

设计 (与 filemem shared_memory.py 对称, 只推不拉):
- 配置: config.yaml plugins.entries.knowledge-base-search.shared_memory
  (同构段; 缺段 / enabled=false -> 整模块 no-op, kb 本地行为字节级不变)
- 只推不拉: 本机 kb 数据源 (唯一权威) 只往外推, 中央 kb_docs 绝不回写
  本地稳定 jsonl 源 (knowledge-base-ingest.jsonl 只被本机写). 中央行视为
  "别人的参考", 只读可查, 不自动注入, 不回写本地.
- 门控简化: 单向后每 doc 只有原 device 能写, 无跨设备冲突. 中央 upsert =
  原设备幂等覆盖 + 非原设备拒写 (token 鉴权 + device 归属).
- 版本存 doc["version"]: 仅防"原设备自己"重放/乱序 (单调), 不做跨设备
  tie-break (那套已随双向语义删除).
- pkl 是派生物, 不入中央, 各机本地重建 (two-layer invariant 不变).

砍掉 (相对 S3-S5 双向版):
- pull 增量腿 (last_pull 水位 / 全量首拉 / 增量拉) — 不拉了
- _columndict_to_doc / _merge_into_local_source / _gate_accepts — 中央行不进
  本地 jsonl 源, 无跨设备门控 (单向无冲突), 无本地镜像门控
保留:
- 推腿: 本机 doc -> 中央 kb_docs (幂等, 带 device 归属 + token 鉴权)
- 检索面: 查中央 kb_docs (只读 SQL), 结果不落本地 / 不重建 pkl

挂点 (kb_service.py 内联, 不是新插件):
- ingest() confirmed 分支: 本机 doc 落盘后
- confirm() 成功分支: moved 进 hot 后
两处各调 kb_shared_memory.notify_confirmed(doc_ids, svc) ——
notify_confirmed 内部只做推腿 (无 pull-merge, 无 pkl 重建)。
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 门 2 共识面 doc_kind 白名单 (owner 定稿 2026-10-22, 对齐 filemem 门 2 语义):
# 只有"通用共享知识"三类进共识召回 (confirmed_only=true / 自动注入面查得到).
# manual / experience 是"本机语境强"门类, 只进门 1 全量沉淀池 (推上中央存档),
# 不进共识召回 — 仅模型显式 confirmed_only=false 翻全量时查得到 (只推不自动注入,
# 可读可参考, 跟 filemem 方式一致).
CONSENSUS_DOC_KINDS: tuple = ("reference", "material", "tutorial")


def _load_shared_memory_config() -> Optional[Dict[str, Any]]:
    """读 plugins.entries.knowledge-base-search.shared_memory 段。

    缺段 / enabled != true / 读 config 失败 -> None (no-op)。

    (a) 直读 config.yaml 文件, 不借 hermes_cli.config / ruamel (owner 定稿
    2026-10-23): 与 filemem shared_memory.py 对称。host tools-python 缺
    ruamel/psycopg2/pyyaml — 宿主加载 kb 插件走 spec_from_file_location,
    hermes_cli.config 的 load_config 读不到 ruamel -> 整条 no-op。改:
      1. ruamel.yaml 可用 -> 用它 (anchor/alias: kb 段 pg: *id001 是 alias,
         只有 ruamel/pyyaml 能解)
      2. pyyaml 可用 -> 用它
      3. 都没有 (host tools python) -> 纯文本最小解析 (kb 段是 alias *id001,
         纯文本解析需回查同文件 &id001 定义取五键值 — filemem 是定义处,
         kb 是引用处, 这里补 anchor 解析)
    读文件路径: 机器级 (owner 定稿 2026-10-23): 1 profile = 1 agent 但差异
    下沉 source_profile 行级列, agent_id 是机器级 (跨机 50 台 = 50 个
    agent_id), 不是 per-profile。读序: HERMES_CONFIG env 优先 -> global
    <HERMES_HOME>/config.yaml -> per-profile (HERMES_PROFILE 有值时兜底)。
    全缺 -> None (no-op, 本地字节级不变)。"""
    raw = _read_config_yaml_text()
    if raw is None:
        return None
    parsed: Optional[Dict[str, Any]] = None
    try:
        from ruamel.yaml import YAML
        yaml_ = YAML(typ="safe")
        parsed = yaml_.load(raw) or {}
    except Exception:
        try:
            import yaml as _pyyaml
            parsed = _pyyaml.safe_load(raw) or {}
        except Exception:
            parsed = _parse_kb_shared_memory_minimal(raw)
    try:
        entries = (parsed or {}).get("plugins", {}).get("entries", {})
        kb = entries.get("knowledge-base-search") or {}
        settings = kb.get("settings") or {}
        # Three read paths, in priority order:
        # 1. Panel-saved scalar keys (.settings.sm_enabled, .settings.sm_agent_id, …)
        #    — the current config_schema shape (independent fields, not a dict).
        # 2. Legacy .settings.shared_memory (dict) — intermediate shape.
        # 3. Legacy .shared_memory (dict) — original direct-entry layout.
        if "sm_enabled" in settings:
            enabled = bool(settings.get("sm_enabled", True))
            if not enabled:
                return None
            agent_id = str(settings.get("sm_agent_id") or "").strip()
            token = str(settings.get("sm_token") or "")
            pg = {
                "host": str(settings.get("sm_pg_host") or ""),
                "port": int(settings.get("sm_pg_port") or 5432),
                "db": str(settings.get("sm_pg_db") or ""),
                "user": str(settings.get("sm_pg_user") or ""),
                "password": str(settings.get("sm_pg_password") or ""),
            }
        else:
            sm = settings.get("shared_memory")
            if not isinstance(sm, dict):
                sm = kb.get("shared_memory")
            if not isinstance(sm, dict) or not sm.get("enabled"):
                return None
            agent_id = str(sm.get("agent_id") or "").strip()
            token = str(sm.get("token") or "")
            pg_raw = sm.get("pg") or {}
            pg = {
                "host": str(pg_raw.get("host") or ""),
                "port": int(pg_raw.get("port") or 5432),
                "db": str(pg_raw.get("db") or ""),
                "user": str(pg_raw.get("user") or ""),
                "password": str(pg_raw.get("password") or ""),
            }
        if not agent_id or not pg["host"] or not pg["db"]:
            return None
        return {
            "agent_id": agent_id,
            "token": token,
            "pg": pg,
        }
    except Exception as e:
        logger.warning(f"kb shared_memory: config parse failed, no-op: {e}")
        return None


def _read_config_yaml_text() -> Optional[str]:
    """定位并读 config.yaml 全文 (owner 定稿 2026-10-23, 机器级 agent_id).

    读序: 1. HERMES_CONFIG env  2. <HERMES_HOME>/config.yaml (global,
    机器级 — multiplex 宿主无 HERMES_PROFILE env, 本机 12 profile 共用一份
    机器级配置配在 global)  3. <HERMES_HOME>/profiles/<name>/config.yaml
    (仅 HERMES_PROFILE 有值且 global 缺时兜底)。全缺 -> None。"""
    import os
    explicit = os.environ.get("HERMES_CONFIG", "").strip()
    if explicit and os.path.isfile(explicit):
        try:
            with open(explicit, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    home = os.environ.get("HERMES_HOME", "").strip()
    if not home:
        return None
    candidates = [f"{home}/config.yaml"]
    profile = os.environ.get("HERMES_PROFILE", "").strip()
    if profile:
        candidates.append(f"{home}/profiles/{profile}/config.yaml")
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                continue
    return None


def _parse_kb_shared_memory_minimal(raw: str) -> Dict[str, Any]:
    """纯文本最小解析: 提取 plugins.entries.knowledge-base-search.shared_memory
    段。kb 段内 pg 是 alias (pg: *id001) — 需在同文件找 &id001 定义处 (在
    filemem.shared_memory.pg) 取五键值补进 pg_block。
    filemem 段是 &id001 定义处, kb 段是 *id001 引用处。"""
    import re
    sm_block: Dict[str, Any] = {}
    pg_block: Dict[str, Any] = {}
    in_plugins = in_entries = in_kb = False
    in_sm = in_pg = False
    _sm_indent = 0
    _entries_indent = 0
    _SM_KEYS = {"enabled", "agent_id", "token"}
    _PG_KEYS = {"host", "port", "db", "user", "password"}
    anchor_pg: Optional[Dict[str, Any]] = None  # &id001 定义处的五键

    def _scalar(v: str):
        s = v.strip()
        if s.lower() in ("true", "yes"):
            return True
        if s.lower() in ("false", "no"):
            return False
        return s

    # pass 1: 抓 &id001 (或任意 &idNNN) 的 pg 定义 (在 filemem 段).
    # 逐行扫: 见到 `pg: &NAME` 进 anchor 采集, 采集 pg 子块内的五键值;
    # 行缩进回退到 pg 键级别以下 -> 离开该 anchor 定义.
    anchor_active = False
    anchor_pg_key_indent = 0
    for line in raw.splitlines():
        stripped = line.rstrip()
        m = re.match(r"^(\s+)pg:\s*(\&)(\S+)\s*$", stripped)
        if m:
            anchor_active = True
            anchor_pg = {}
            anchor_pg_key_indent = len(m.group(1))
            continue
        if anchor_active:
            pm = re.match(r"^(\s+)(\S+):(.*)$", stripped)
            if pm:
                k, rest, p_indent = pm.group(1), pm.group(2).strip(), len(pm.group(1))
                if p_indent > anchor_pg_key_indent and k in _PG_KEYS and rest:
                    anchor_pg[k] = _scalar(rest)
                elif p_indent <= anchor_pg_key_indent and k not in _PG_KEYS:
                    anchor_active = False
    for line in raw.splitlines():
        stripped = line.rstrip()
        if not in_plugins:
            if re.match(r"^plugins:\s*$", stripped):
                in_plugins = True
            continue
        if not in_entries:
            m = re.match(r"^(\s+)entries:\s*$", stripped)
            if m:
                in_entries = True
                _entries_indent = len(m.group(1))
            continue
        if not in_kb:
            m = re.match(r"^(\s+)(\S+):\s*$", stripped)
            if m and len(m.group(1)) == _entries_indent + 2 and m.group(2) == "knowledge-base-search":
                in_kb = True
            continue
        m = re.match(r"^(\s+)(\S+):(.*)$", stripped)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        if not in_sm:
            if key == "shared_memory" and not rest.strip():
                in_sm = True
                _sm_indent = indent
                continue
            if indent <= _entries_indent + 2:
                break
            continue
        # in_sm
        if in_pg:
            if key in _PG_KEYS and rest.strip():
                pg_block[key] = _scalar(rest)
            elif key not in _PG_KEYS and indent <= _sm_indent + 2:
                in_pg = False
            continue
        if key in _SM_KEYS and rest.strip():
            sm_block[key] = _scalar(rest)
        elif key == "pg" and not pg_block:
            # alias 引用 (*id001): 用 pass 1 抓的 anchor 定义补
            in_pg = True
            if anchor_pg:
                pg_block.update(anchor_pg)
        if indent <= _sm_indent and key not in _SM_KEYS and key != "pg":
            break
    if not in_sm:
        return {}
    sm_block.setdefault("pg", pg_block)
    return {"plugins": {"entries": {"knowledge-base-search": {"shared_memory": sm_block}}}}


def _ensure_pg_dep_paths() -> None:
    """Make psycopg2 importable from the running interpreter.

    Superseded by kbs_dep_bootstrap.ensure_pg_dep_paths (shared single source of
    truth, 2026-11-11) — kept as a thin re-export wrapper so any existing
    caller/referencer of this exact name still resolves; the actual splice
    logic now lives in kbs_dep_bootstrap."""
    import kbs_dep_bootstrap
    kbs_dep_bootstrap.ensure_pg_dep_paths()


_ensure_pg_dep_paths()


class _PgConn:
    """最小 PG 连接 (与 filemem 侧同形态, 两仓各一份 —— X 方案代价)。"""

    def __init__(self, pg: Dict[str, Any]):
        import psycopg2
        self._conn = psycopg2.connect(
            host=pg["host"], port=int(pg["port"]), dbname=pg["db"],
            user=pg["user"], password=pg["password"],
            connect_timeout=5,
        )
        self._conn.autocommit = False

    def cur(self):
        return self._conn.cursor()

    def commit(self):
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


class _TokenRejected(Exception):
    """中央 agent_registry 拒 (unknown/revoked)。S4 owner 拍 (a): 校验在
    中央 SQL 函数层 require_agent, 客户端区分"拒"与"连不上"。"""
    pass


def _central_auth(conn, agent_id: str, token: str) -> None:
    import hashlib
    token_hash = hashlib.sha256((token or "").encode("utf-8")).hexdigest()
    try:
        conn.cur().execute("SELECT require_agent(%s, %s)", (agent_id, token_hash))
    except Exception as e:
        raise _TokenRejected(str(e)) from e


# ── kb doc -> 中央 kb_docs 列 映射 (version 存 doc["version"], 仅防重放) ───

def _doc_to_columndict(doc: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    """kb doc -> 中央 kb_docs 列 dict。
    version_ts = updated_at / occurred_at 任一非空即取 (全缺才 None -> 中央
    COALESCE now() 兜底)。version 仅防原设备重放, 不做跨设备门控。"""
    meta = {k: doc.get(k) for k in ("doc_kind", "origin_kind", "premise_status",
                                    "topic", "task_type", "premise") if doc.get(k) is not None}
    ver_ts = doc.get("updated_at") or doc.get("occurred_at")
    return {
        "docid": doc.get("docid"),
        "title": doc.get("title", ""),
        "content": doc.get("content", ""),
        "source_profile": doc.get("source_profile"),
        "agent_id": agent_id,
        "version": int(doc.get("version") or 1),
        "version_ts": ver_ts,
        "occurred_at": doc.get("occurred_at"),
        "status": doc.get("status") or "confirmed",
        "doc_kind": doc.get("doc_kind"),
        "meta": meta,
    }


# ── 主入口: doc 落盘后调用 (只推, 不拉) ─────────────────────────────────

_local_lock = threading.Lock()


def notify_confirmed(svc, new_docids: List[str]) -> Dict[str, Any]:
    """svc = KbSearchService 实例 (用其 _data_dir)。
    new_docids = 本次落盘的 docid (confirmed+unconfirmed 都推门 1 沉淀池;
    空 -> no-op)。只推不拉: 无 pull / 无本地 jsonl 回写 / 无 pkl 重建。
    返回 {"status": "ok"|"skipped"|"rejected", "pushed": n}"""
    cfg = _load_shared_memory_config()
    if cfg is None:
        return {"status": "skipped", "reason": "config-absent-or-disabled"}
    if not new_docids:
        return {"status": "skipped", "reason": "no-ids", "pushed": 0}
    with _local_lock:
        try:
            return _push_locked(svc, cfg, new_docids)
        except _TokenRejected as e:
            logger.warning(f"kb shared_memory: central token rejected for agent "
                           f"{cfg.get('agent_id')}: {e}")
            return {"status": "rejected", "reason": "token-rejected"}
        except Exception as e:
            logger.warning(f"kb shared_memory notify no-op (fail-soft): {e}")
            return {"status": "skipped", "reason": f"fail-soft: {e}"}


def _push_locked(svc, cfg: Dict[str, Any], new_docids: List[str]) -> Dict[str, Any]:
    conn = _PgConn(cfg["pg"])
    try:
        agent_id = cfg["agent_id"]
        _central_auth(conn, agent_id, cfg.get("token", ""))
        cur = conn.cur()
        import hashlib
        token_hash = hashlib.sha256((cfg.get("token") or "").encode("utf-8")).hexdigest()
        # 只推: 本机落盘 doc -> 中央 kb_docs (全量, 幂等 upsert)。
        # 无 pull 腿 / 无水位 / 无本地 jsonl 回写 / 无 pkl 重建。
        pushed = 0
        for did in new_docids:
            doc = _read_local_doc(svc, did)
            if doc is None:
                continue
            cd = _doc_to_columndict(doc, agent_id)

            def _js(v):
                from datetime import datetime as _dt
                if isinstance(v, _dt):
                    return v.isoformat()
                if isinstance(v, str) and v and "T" in v:
                    try:
                        return _dt.fromisoformat(v.replace("Z", "+00:00")).isoformat()
                    except Exception:
                        return None  # 非时间戳串 -> 中央 COALESCE now() 兜底
                return v
            cur.execute(
                "SELECT upsert_kb_doc(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (cd["docid"], cd["title"], cd["content"], cd["source_profile"],
                 cd["agent_id"], cd["version"], _js(cd["version_ts"]),
                 _js(cd["occurred_at"]), cd["status"], cd["doc_kind"],
                 json.dumps(cd["meta"], ensure_ascii=False,
                            default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)),
                 token_hash),
            )
            pushed += 1
        conn.commit()
        return {"status": "ok", "pushed": pushed}
    finally:
        conn.close()


# ── Q1 启动对账腿 (owner 定稿 2026-10-23, 选 A: _svc() 构造后挂, 模块级哨兵) ──

_svc_reconciled = False  # 模块级 host 生命周期哨兵 (kb 是 hook/工具型插件,
                         # _svc() 每 session 可能重入, 哨兵挂模块防 per-session 重置)


def _all_local_docids(svc) -> List[str]:
    """本地 hot+cold pkl 双快照 docid 全集 (对账基准源 = pkl, owner 定稿)."""
    ids = set()
    try:
        for d in (svc._hot.docs_snapshot() + svc._cold.docs_snapshot()):
            if d.get("docid"):
                ids.add(str(d["docid"]))
    except Exception:
        pass
    return sorted(ids)


def reconcile_local_to_central(svc) -> Dict[str, Any]:
    """Q1 启动对账腿: 本地 hot+cold 全集 docid vs 中央 kb_docs 差集补推
    (幂等, 只推本地有中央无的, 根除重启/宕机期间 notify_confirmed 没 fire 的
    缺口). 与 filemem reconcile_local_to_central 同型. fail-soft: 中央不可达/
    token 拒静默跳过. 返回 {"status", "missing", "pushed"}.

    基准源 = pkl 双快照 (非 staging_docs.jsonl): 它是本地权威源 (格式 B jsonl
    是跨机导出格式, 本机对账用 pkl 零新依赖)."""
    cfg = _load_shared_memory_config()
    if cfg is None:
        return {"status": "skipped", "reason": "config-absent-or-disabled",
                "missing": 0, "pushed": 0}
    conn = _PgConn(cfg["pg"])
    try:
        agent_id = cfg["agent_id"]
        _central_auth(conn, agent_id, cfg.get("token", ""))
        cur = conn.cur()
        cur.execute(
            "SELECT docid FROM kb_docs WHERE agent_id = %s", (agent_id,))
        central_ids = {str(r[0]) for r in cur.fetchall()}
        local_ids = _all_local_docids(svc)
        missing = [d for d in local_ids if d not in central_ids]
        if not missing:
            return {"status": "ok", "missing": 0, "pushed": 0,
                    "reason": "in-sync"}
        with _local_lock:
            result = _push_locked(svc, cfg, missing)
        return {"status": result.get("status", "ok"),
                "missing": len(missing), "pushed": result.get("pushed", 0)}
    except _TokenRejected as e:
        logger.warning(f"kb shared_memory reconcile token rejected: {e}")
        return {"status": "rejected", "reason": "token-rejected",
                "missing": 0, "pushed": 0}
    except Exception as e:
        logger.warning(f"kb shared_memory reconcile no-op (fail-soft): {e}")
        return {"status": "skipped", "reason": f"fail-soft: {e}",
                "missing": 0, "pushed": 0}
    finally:
        conn.close()


def _svc_reconcile_once(svc) -> Dict[str, Any]:
    """_svc() 构造后调用: 模块级哨兵守, host 生命周期只跑一次对账."""
    global _svc_reconciled
    if _svc_reconciled:
        return {"status": "skipped", "reason": "already-reconciled",
                "missing": 0, "pushed": 0}
    result = reconcile_local_to_central(svc)
    # P1 fix (reviewer 复盘 2026-10-23): 只有非 fail-soft 结果才置哨兵。
    # fail-soft (中央不可达/token 拒) 不置 _svc_reconciled, 留给下次 _svc()
    # 重入重试 — 与 filemem Q1 "失败不置 flag" 语义对齐. 中央长期不可达时
    # 每个 _svc() 重入都重试 (对账是幂等差集查询, 成本可控); 中央可达时
    # 一次置位, host 生命周期不再重跑.
    if not str(result.get("reason", "")).startswith("fail-soft"):
        _svc_reconciled = True
    if result.get("pushed", 0) > 0:
        logger.info(f"kb shared_memory: startup reconcile pushed "
                    f"{result['pushed']} missing docs to central "
                    f"(gap from pre-restart notify_confirmed)")
    return result


def _read_local_doc(svc, did: str) -> Optional[Dict[str, Any]]:
    """从本地 confirmed corpus (hot + cold snapshot) 读 doc。"""
    try:
        for d in (svc._hot.docs_snapshot() + svc._cold.docs_snapshot()):
            if str(d.get("docid")) == str(did):
                return d
    except Exception:
        return None
    return None


def bump_version(doc: Dict[str, Any]) -> int:
    """confirmed doc 变更 (ingest/confirm/supersede 后重进 hot) 时: 单调 +1。
    新 doc = 1, 变更 = 旧 + 1。仅防原设备重放, 不做跨设备门控。"""
    old = int(doc.get("version") or 0)
    return old + 1


# ── 检索面: 查中央 (只读, 结果不落本地 / 不重建 pkl) ───────────────────

def search_central(cfg: Dict[str, Any], query: str,
                   confirmed_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
    """查中央 kb_docs (别人的参考知识)。只读, 返回 doc dict 列表, 不写本地。
    confirmed_only=True -> 查共识 (status='confirmed'); False -> 查全量沉淀。
    结果由调用方决定用不用, 绝不自动进本地 corpus / 重建 pkl。"""
    conn = _PgConn(cfg["pg"])
    try:
        agent_id = cfg["agent_id"]
        _central_auth(conn, agent_id, cfg.get("token", ""))
        cur = conn.cur()
        cols = ("SELECT docid, title, content, source_profile, agent_id, version, "
                "version_ts, occurred_at, status, doc_kind, meta FROM kb_docs")
        sql = cols + " WHERE (title ILIKE %s OR content ILIKE %s OR meta::text ILIKE %s)"
        params: List[Any] = [f"%{query}%", f"%{query}%", f"%{query}%"]
        if confirmed_only:
            sql += " AND status = 'confirmed'"
            # 门 2 共识面 (owner 定稿 2026-10-22): 只放行 CONSENSUS_DOC_KINDS 三类
            # (reference/material/tutorial 通用共享知识). manual/experience 本机语境强,
            # 不进共识召回 — 只在全量面 (confirmed_only=False) 查得到.
            # doc_kind 为 NULL (agent 自动捕获未打标) 不进共识面 (宁缺不误灌).
            _placeholders = ", ".join(["%s"] * len(CONSENSUS_DOC_KINDS))
            sql += f" AND doc_kind IN ({_placeholders})"
            params.extend(CONSENSUS_DOC_KINDS)
        sql += " ORDER BY version_ts DESC NULLS LAST LIMIT %s"
        params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
        def _ts(v):
            """timestamptz -> ISO string (json.dumps 前必须去掉 datetime 对象)."""
            if v is None:
                return None
            if hasattr(v, "isoformat"):
                return v.isoformat()
            return v
        return [
            dict(zip(["docid", "title", "content", "source_profile", "agent_id",
                      "version", "version_ts", "occurred_at", "status", "doc_kind",
                      "meta"],
                    [r[0], r[1], r[2], r[3], r[4], r[5],
                     _ts(r[6]), _ts(r[7]), r[8], r[9], r[10]]))
            for r in rows
        ]
    except _TokenRejected:
        return []
    except Exception as e:
        logger.warning(f"kb shared_memory search_central no-op (fail-soft): {e}")
        return []
    finally:
        conn.close()
