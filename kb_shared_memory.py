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
        sql += " ORDER BY version_ts DESC NULLS LAST LIMIT %s"
        params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [
            dict(zip(["docid", "title", "content", "source_profile", "agent_id",
                      "version", "version_ts", "occurred_at", "status", "doc_kind",
                      "meta"], r)) for r in rows
        ]
    except _TokenRejected:
        return []
    except Exception as e:
        logger.warning(f"kb shared_memory search_central no-op (fail-soft): {e}")
        return []
    finally:
        conn.close()
