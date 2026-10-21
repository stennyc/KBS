"""S3: kb 侧内联共享同步 (owner 定稿 2026-10-21, PRD /root/.hermes/docs/shared-memory-prd.md)

kb 与 filemem 对称的内联 B' 同步 (X 方案, 两仓各一份, 门控逻辑同语义)。
中央库 = 同一台 Postgres 的 kb_docs 表 (version 门控, 与 filemem memories 表同构)。

设计 (与 filemem shared_memory.py 对称):
- 配置: config.yaml plugins.entries.knowledge-base-search.shared_memory
  (同构段; 缺段 / enabled=false -> 整模块 no-op, kb 本地行为字节级不变)
- 门控: kb doc 加 version (本地单调, 与 filemem shared_version 同一套语义)
  大者赢 -> 同 version 比 version_ts -> 再比 agent_id 字典序小者赢;
  被覆盖旧版进 memory_history (中央 SQL 层 upsert_kb_doc 已记)。
- 检索 = B' 写前拉后推: 只在"confirmed doc 进 hot"时刻 (ingest confirmed
  分支 + confirm() staging->hot 移) pull 中央 kb_docs 增量 -> 门控合入本地
  稳定 jsonl 源 (knowledge-base-ingest.jsonl, 不是 pkl) -> 本地重建 pkl
  (two-layer invariant: pkl 是派生物, 不入中央, 各机本地重建)。
- fail-soft: PG 不可达 / config 缺失 / psycopg2 缺失 -> 静默跳过, 绝不
  阻塞 kb 本地写路径。

挂点 (kb_service.py 内联, 不是新插件):
- ingest() confirmed 分支: self._backend.update(confirmed) 之后
- confirm() 成功分支: moved 移进 hot 之后
两处各调 kb_shared_memory.notify_confirmed(doc_ids, svc) ——
notify_confirmed 内部做 pull-merge-push + 本地 jsonl 源回写 + pkl 重建。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_shared_memory_config() -> Optional[Dict[str, Any]]:
    """读 plugins.entries.knowledge-base-search.shared_memory 段。
    缺段 / enabled != true / 读 config 失败 -> None (no-op)。"""
    try:
        from hermes_cli.config import load_config, load_config_readonly
    except Exception:
        return None
    try:
        config = load_config_readonly()
    except Exception:
        try:
            config = load_config()
        except Exception:
            return None
    try:
        entries = config.get("plugins", {}).get("entries", {})
        kb = entries.get("knowledge-base-search") or {}
        sm = kb.get("shared_memory")
        if not isinstance(sm, dict) or not sm.get("enabled"):
            return None
        pg = sm.get("pg") or {}
        agent_id = str(sm.get("agent_id") or "").strip()
        if not agent_id or not pg.get("host") or not pg.get("db"):
            return None
        return {
            "agent_id": agent_id,
            "token": str(sm.get("token") or ""),
            "pg": {
                "host": str(pg.get("host")),
                "port": int(pg.get("port") or 5432),
                "db": str(pg.get("db")),
                "user": str(pg.get("user")),
                "password": str(pg.get("password") or ""),
            },
        }
    except Exception as e:
        logger.warning(f"kb shared_memory: config read failed, no-op: {e}")
        return None


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


# ── kb doc <-> 中央 kb_docs 列 映射 (version 存 doc["version"]) ───────────

def _doc_to_columndict(doc: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    """kb doc -> 中央 kb_docs 列 dict。
    S5 双审 critical 修复: version_ts 不再单取 doc['updated_at'] (kb doc 从没有
    该键 -> 恒 None -> 中央 kb_docs.version_ts 走 now() 兜底, 门控第二层
    version_ts 比较形同虚设, 全靠 agent_id 封顶)。改为 updated_at / occurred_at /
    version_ts 三个键任一非空即取, 全缺才 None (中央 COALESCE now() 兜底)。
    与 filemem 侧 _row_to_rowdict 的 updated_at 语义对称。"""
    meta = {k: doc.get(k) for k in ("doc_kind", "origin_kind", "premise_status",
                                    "topic", "task_type", "premise") if doc.get(k) is not None}
    ver_ts = doc.get("updated_at") or doc.get("occurred_at") or doc.get("version_ts")
    return {
        "docid": doc.get("docid"),
        "title": doc.get("title", ""),
        "content": doc.get("content", ""),
        "source_profile": doc.get("source_profile"),
        "agent_id": agent_id,
        "version": int(doc.get("version") or 1),
        "version_ts": ver_ts,
        "occurred_at": doc.get("occurred_at"),
        "status": "confirmed",
        "doc_kind": doc.get("doc_kind"),
        "meta": meta,
    }


def _columndict_to_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    """中央 kb_docs 行 -> 本地稳定 jsonl 源 doc (datetime -> ISO 字符串,
    与 filemem 侧 _rowdict_to_local 同款处理; pkl 重建读的是 jsonl 源)。"""
    from datetime import datetime as _dt

    def _ts(v):
        return v.isoformat() if isinstance(v, _dt) else v

    meta = d.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    doc = {
        "docid": d.get("docid"),
        "title": d.get("title"),
        "content": d.get("content", ""),
        "source_profile": d.get("source_profile"),
        "version": int(d.get("version") or 1),
        "status": "confirmed",
        "updated_at": _ts(d.get("version_ts")),
        "occurred_at": _ts(d.get("occurred_at")),
        "doc_kind": d.get("doc_kind"),
    }
    doc.update({k: _ts(v) for k, v in meta.items()})
    return doc


# ── 本地门控 (与中央 upsert_kb_doc PL/pgSQL 同语义, 两仓各一份) ──────────

def _gate_accepts(
    cur_doc: Optional[Dict[str, Any]],
    new_doc: Dict[str, Any], new_agent_id: str,
) -> bool:
    """本地已有 cur_doc 时, 中央拉回的 new_doc 是否胜出。
    version 大者赢 -> 同 version 比 updated_at -> 再比 agent_id 字典序小者赢。
    cur_doc 缺 version (本地存量未回填) 视同 0, 新行必赢 (回填后不会发生)。

    S5 双审 critical 修复: 时间戳比较统一走 epoch (datetime.fromisoformat),
    不再用 ISO 字符串字典序 (时区/微秒位数差异让"同一瞬时"判为不等, 与中央
    TIMESTAMPTZ 瞬时比较 diverge = split-brain, 与 filemem 侧同修)。"""
    from datetime import datetime as _dt
    def _epoch(v):
        if v is None:
            return None
        try:
            if isinstance(v, _dt):
                return v.timestamp()
            s = str(v)
            if not s:
                return None
            d = _dt.fromisoformat(s.replace("Z", "+00:00"))
            if d.tzinfo is None:
                from datetime import timezone as _tz
                d = d.replace(tzinfo=_tz.utc)
            return d.timestamp()
        except Exception:
            return None
    if cur_doc is None:
        return True
    cur_v = int(cur_doc.get("version") or 0)
    new_v = int(new_doc.get("version") or 1)
    if new_v > cur_v:
        return True
    if new_v == cur_v:
        NEG = float("-inf")
        c = _epoch(cur_doc.get("updated_at"))
        c = c if c is not None else NEG
        n = _epoch(new_doc.get("updated_at"))
        n = n if n is not None else NEG
        if n > c:
            return True
        if n == c:
            return new_agent_id < str(cur_doc.get("_remote_agent_id") or "")
    return False


# ── 本地稳定 jsonl 源回写 (拉回的增量 -> knowledge-base-ingest.jsonl) ─────

_KB_INGEST_JSONL = "knowledge-base-ingest.jsonl"


def _merge_into_local_source(kb_data_dir: Path, incoming: List[Dict[str, Any]],
                             agent_id: str) -> int:
    """把中央拉回的 kb_docs 增量按门控合入本地稳定 jsonl 源。
    返回本地真被覆盖的 doc 数。jsonl 每行 = 一个 confirmed doc。"""
    if not incoming:
        return 0
    src = kb_data_dir / _KB_INGEST_JSONL
    local: Dict[str, Dict[str, Any]] = {}
    if src.exists():
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                local[str(d.get("docid"))] = d
    clobbered = 0
    for d in incoming:
        doc = _columndict_to_doc(d)
        doc["_remote_agent_id"] = d.get("agent_id")
        cur = local.get(str(doc.get("docid")))
        if _gate_accepts(cur, doc, agent_id):
            doc.pop("_remote_agent_id", None)
            local[str(doc["docid"])] = doc
            clobbered += 1
    if clobbered:
        tmp = src.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for doc in local.values():
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        os.replace(tmp, src)
    return clobbered


# ── 主入口: confirmed doc 进 hot 后调用 (写前拉后推) ──────────────────────

_local_lock = threading.Lock()


def notify_confirmed(svc, new_confirmed_docids: List[str]) -> Dict[str, Any]:
    """svc = KbSearchService 实例 (用其 _data_dir)。
    new_confirmed_docids = 本次刚进 confirmed corpus 的 docid。
    返回 {"status": "skipped"|"ok", "pulled": n, "pushed": n, "clobbered_local": n}"""
    cfg = _load_shared_memory_config()
    if cfg is None:
        return {"status": "skipped", "reason": "config-absent-or-disabled"}
    with _local_lock:
        try:
            return _notify_locked(svc, cfg, new_confirmed_docids)
        except _TokenRejected as e:
            logger.warning(f"kb shared_memory: central token rejected for agent "
                           f"{cfg.get('agent_id')}: {e}")
            return {"status": "rejected", "reason": "token-rejected"}
        except Exception as e:
            logger.warning(f"kb shared_memory notify no-op (fail-soft): {e}")
            return {"status": "skipped", "reason": f"fail-soft: {e}"}


def _notify_locked(svc, cfg: Dict[str, Any], new_confirmed_docids: List[str]) -> Dict[str, Any]:
    conn = _PgConn(cfg["pg"])
    try:
        agent_id = cfg["agent_id"]
        # S4 owner 拍 (a): 中央层 token 硬约束 (与 filemem 侧对称)
        _central_auth(conn, agent_id, cfg.get("token", ""))
        kind = "kb"
        cur = conn.cur()

        # 1) 水位线
        cur.execute(
            "SELECT last_pull FROM agent_watermarks WHERE agent_id=%s AND kind=%s",
            (agent_id, kind))
        row = cur.fetchone()
        since = row[0] if row else None

        # 2) pull 增量 kb_docs
        cols = ("docid, title, content, source_profile, agent_id, version, "
                "version_ts, occurred_at, status, doc_kind, meta")
        if since is None:
            cur.execute(
                f"SELECT {cols} FROM kb_docs WHERE status='confirmed' "
                "ORDER BY version_ts LIMIT 5000")
        else:
            # psycopg2 把 since (last_pull, TIMESTAMPTZ) 读回 datetime; 若上游
            # 传入 str 需转 datetime (与 filemem 侧对称, 防 datetime>str 崩)
            from datetime import datetime as _dtc, timezone as _tzc
            if isinstance(since, str):
                since = _dtc.fromisoformat(since.replace("Z", "+00:00"))
            if since.tzinfo is None:
                since = since.replace(tzinfo=_tzc.utc)
            cur.execute(
                f"SELECT {cols} FROM kb_docs WHERE status='confirmed' "
                "AND version_ts > %s ORDER BY version_ts LIMIT 5000", (since,))
        raw = cur.fetchall()
        incoming = [
            dict(zip(["docid", "title", "content", "source_profile", "agent_id",
                      "version", "version_ts", "occurred_at", "status", "doc_kind",
                      "meta"], r)) for r in raw
        ]

        # 3) 门控合入本地 jsonl 稳定源 (pkl 由 kb 既有 rebuild 流程重建)
        clobbered = _merge_into_local_source(Path(svc._data_dir), incoming, agent_id)

        # 4) push 本机新 confirmed doc (走中央 upsert_kb_doc, 门控在 SQL 层)
        pushed = 0
        for did in new_confirmed_docids:
            doc = _read_local_doc(svc, did)
            if doc is None:
                continue
            cd = _doc_to_columndict(doc, agent_id)

            def _js(v):
                from datetime import datetime as _dt
                return _dt.fromisoformat(v).isoformat() if isinstance(v, str) and v and "T" in v else v
            cur.execute(
                "SELECT upsert_kb_doc(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (cd["docid"], cd["title"], cd["content"], cd["source_profile"],
                 cd["agent_id"], cd["version"], _js(cd["version_ts"]),
                 _js(cd["occurred_at"]), cd["status"], cd["doc_kind"],
                 json.dumps(cd["meta"], ensure_ascii=False,
                            default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o))),
            )
            pushed += 1

        # 5) 推水位线
        if incoming:
            last_ts = max((d["version_ts"] for d in incoming if d.get("version_ts")),
                          default=None)
            if last_ts:
                cur.execute(
                    "INSERT INTO agent_watermarks (agent_id, kind, last_pull, last_push, last_version_seen) "
                    "VALUES (%s, %s, %s, now(), (SELECT max(version) FROM kb_docs)) "
                    "ON CONFLICT (agent_id, kind) DO UPDATE SET "
                    "last_pull = EXCLUDED.last_pull, last_push = now(), "
                    "last_version_seen = GREATEST(agent_watermarks.last_version_seen, EXCLUDED.last_version_seen)",
                    (agent_id, kind, last_ts))
        conn.commit()
        return {"status": "ok", "pulled": len(incoming), "pushed": pushed,
                "clobbered_local": clobbered}
    finally:
        conn.close()


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
    新 doc = 1 (存量回填后起步 1), 变更 = 旧 + 1。与 filemem 侧同语义。"""
    old = int(doc.get("version") or 0)
    return old + 1
