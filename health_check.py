#!/usr/bin/env python3
"""知识库检索服务健康检查（上线后可观测）

执行要求（重要——诊断工具自身可被误读）：
- 必须用 pin 解释器跑（KB_PY_BIN，默认 /root/.hermes/tools/python-3.14.7+20260901-linux-x64/bin/python3.14，
  即 7 个 gateway profile 加载 plugin 时使用的解释器；该解释器装有
  rank_bm25 + jieba）。用错的解释器会让本检查 import 即失败
  （ModuleNotFoundError）→ 检查报红但 KB 服务并未挂——这是解释器
  误读，不是服务故障。
- 报红时先确认两件事再下"服务挂了"的结论：
  (1) 跑的是否 pin 解释器（[0] 判据）；
  (2) gateway 判据取的是否权威信号（systemctl --user is-active，
      ps 子串 'tui_gateway' 已废弃——gateway argv 实为 '-c gateway
      run'，grep 必假阴）。

Run:  /root/.hermes/tools/python-3.14.7+20260901-linux-x64/bin/python3.14 knowledge-base/health_check.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kbs_dep_bootstrap import resolve_hermes_home, resolve_kb_data_dir  # noqa: E402

# runbook 唯一入口三件套：pin 解释器 / 共享数据目录 / 共享 로그（与 plugin
# 适配器同一套 env 约定：KB_PY_BIN / KB_DATA_DIR / KB_LOG_PATH）。
# pin 解释器优先从 KB_PY_BIN 获取，fallback 到 $HERMES_HOME/tools/ 下的 python，
# 再 fallback 到 sys.executable（当前解释器）。
_hermes_home = resolve_hermes_home()
if sys.platform == "win32":
    _DEFAULT_PY_BIN = os.path.join(_hermes_home, "tools", "python.exe")
    if not os.path.exists(_DEFAULT_PY_BIN):
        _DEFAULT_PY_BIN = sys.executable
else:
    # Linux: 尝试常见 tools 目录， fallback 到 sys.executable
    import glob
    _candidates = sorted(glob.glob(os.path.join(_hermes_home, "tools", "*", "bin", "python3.14")))
    _DEFAULT_PY_BIN = _candidates[0] if _candidates else sys.executable
PIN_PY_BIN = os.environ.get("KB_PY_BIN", _DEFAULT_PY_BIN)
# Shared corpus dir: AUTO-DETECTS to ONE host tree (KB_DATA_DIR pin >
# platform-native home > HERMES_HOME) — the 2026-11-17 split-brain fix, so
# the runbook probes the SAME corpus the plugin adapter writes.
DATA_DIR = Path(resolve_kb_data_dir())
LOG_PATH = os.environ.get(
    "KB_LOG_PATH", os.path.join(_hermes_home, "logs", "knowledge-base.log"),
)
PROFILES = ["coder", "orchestrator", "reviewer", "researcher",
            "product-manager", "data-engineer", "default"]

# 服务 import 保护：走错解释器（无 rank_bm25/jieba）时，本文件顶部的
# kb_service import 会炸在 check() 之前——把"炸"收进 _IMPORT_ERR，让
# [0] 段输出可读的红，而不是裸 traceback。
try:
    from kb_service import KbSearchService, configure_logging
    _IMPORT_ERR = ""
except Exception as e:  # ModuleNotFoundError = 大概率解释器不对
    KbSearchService = None
    configure_logging = None
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _check_interpreter_pin() -> bool:
    """[0] 解释器 pin：当前进程必须就是 pin 解释器，且 pin 解释器依赖齐。"""
    print("\n[0] 解释器 pin（KB_PY_BIN）")
    pin = PIN_PY_BIN
    if not os.path.exists(pin):
        print(f"  ✗ pin 解释器路径不存在: {pin}")
        print("  （KB_PY_BIN 配错或该解释器被移走；核对部署环境后重跑）")
        return False
    r = subprocess.run(
        [pin, "-c", "import rank_bm25, jieba"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  ✗ pin 解释器缺 rank_bm25/jieba: {r.stderr.strip()[:120]}")
        return False
    print(f"  ✓ pin: {pin}（rank_bm25 + jieba 可 import）")
    if os.path.realpath(pin) != os.path.realpath(sys.executable):
        print(f"  ✗ 当前进程跑在 {sys.executable}（非 pin）——本检查绿/红不可信")
        print(f"  → 请用 pin 解释器重跑: {pin} {Path(__file__).name}")
        return False
    print("  ✓ 当前进程 = pin 解释器")
    return True


def _profile_injected(config: Path) -> tuple[bool, str]:
    """[7] YAML 结构判定（非裸子串）：plugins.enabled 含
    knowledge-base-search 且不在 plugins.disabled 里（disabled 命中 =
    假注入，也报红）。"""
    data = None
    try:
        from ruamel.yaml import YAML
        data = YAML().load(config.open())
    except ModuleNotFoundError:
        import yaml
        data = yaml.safe_load(config.read_text())
    if data is None:
        return False, "config 为空"
    plugins = data.get("plugins") or {}
    enabled = plugins.get("enabled") or []
    disabled = plugins.get("disabled") or []
    if isinstance(enabled, str):
        enabled = [enabled]
    if isinstance(disabled, str):
        disabled = [disabled]
    if "knowledge-base-search" in enabled \
            and "knowledge-base-search" not in disabled:
        return True, "enabled"
    if "knowledge-base-search" in disabled:
        return False, "在 disabled 列表（假注入）"
    return False, "enabled 缺 knowledge-base-search"


def _gateway_active() -> tuple[bool, str]:
    """[8] gateway 权威判据：systemctl --user is-active（ops 备忘同款
    信号）；非 systemd 环境（dev/CI）降级 diag log 末条 gateway.start。"""
    try:
        r = subprocess.run(
            ["systemctl", "--user", "is-active", "hermes-gateway.service"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            state = r.stdout.strip()
            return state == "active", f"systemctl --user is-active = {state}"
        err = (r.stderr or r.stdout).strip().splitlines()
        err = err[-1] if err else f"rc={r.returncode}"
    except (OSError, subprocess.SubprocessError):
        err = "systemctl 不可用"
    # 降级：diag log 末条 gateway.start（只证最近一次启动，不证运行中）
    log = Path(os.environ.get("KB_DIAG_LOG_PATH", os.path.join(_hermes_home, "logs", "gateway-exit-diag.log")))
    if log.exists():
        last = [l for l in log.read_text().splitlines()
                if "gateway.start" in l]
        if last:
            return False, (f"{err}；降级: diag log 末条 gateway.start="
                           f"{last[-1].strip()[:80]}...（降级信号不算运行中）")
    return False, f"{err}；diag log 亦无 gateway.start"


def check() -> bool:
    ok = True
    # [0] 必须先于一切——走错解释器时后续全不可信
    ok = _check_interpreter_pin() and ok
    if not ok or _IMPORT_ERR:
        print("\n[!] 服务 import 失败（疑似解释器不对）")
        if _IMPORT_ERR:
            print(f"  ✗ {_IMPORT_ERR}")
            print("  用 pin 解释器重跑（见文件头 Run 行）")
        print("\n" + "=" * 60)
        print("[FAIL] 健康检查未通过（解释器层，KB 服务状态未知）")
        print("=" * 60)
        return False

    print("\n" + "=" * 60)
    print("知识库检索服务健康检查")
    print("=" * 60)

    # [1] 数据目录
    print("\n[1] 数据目录")
    if DATA_DIR.exists():
        items = sorted(p.name for p in DATA_DIR.iterdir())
        print(f"  ✓ {DATA_DIR} 存在")
        print(f"  内容: {items}")
    else:
        print(f"  ✗ {DATA_DIR} 不存在（部署前置未跑：deploy_knowledge_base_plugin.sh --apply）")
        return False

    # [2] 软链
    print("\n[2] Plugin 软链")
    link = Path(os.path.join(_hermes_home, "plugins", "knowledge-base-search"))
    if link.is_symlink():
        target = link.resolve()
        print(f"  ✓ 软链有效: {target}")
        if not (target / "__init__.py").exists():
            print("  ✗ 插件代码缺失")
            return False
    else:
        print("  ✗ 软链不存在或无效（应为指向 repo 的 ln -s，非拷贝）")
        return False

    # [3] 构造 service（启动守卫 loud fail；顺手接上共享日志）
    print("\n[3] Service 构造")
    try:
        configure_logging(LOG_PATH)
        svc = KbSearchService(DATA_DIR)
        print("  ✓ Service 构造成功")
    except Exception as e:
        print(f"  ✗ Service 构造失败: {e}")
        return False

    # [4] 热层状态
    print("\n[4] 热层状态")
    print(f"  文档数: {len(svc._hot.docs_snapshot())}")
    print(f"  Generation: {svc._hot.generation}")

    # [5] 冷层状态
    print("\n[5] 冷层状态")
    print(f"  文档数: {len(svc._cold.docs_snapshot())}")
    print(f"  Generation: {svc._cold.generation}")

    # [6] 搜索冒烟
    print("\n[6] 搜索冒烟")
    try:
        result = svc.search("测试", top_k=1)
        status = result.get("status", "unknown")
        total = result.get("data", {}).get("total", -1)
        print(f"  查询 '测试': status={status}, total={total}")
        if status != "success":
            print("  ⚠ 搜索异常（status 非 success，查 kb.log + seq）")
    except Exception as e:
        print(f"  ✗ 搜索异常: {e}")
        return False

    # [7] Profile 注入（YAML 结构判定）
    print("\n[7] Profile 注入检查（YAML parse）")
    all_injected = True
    for p in PROFILES:
        config = Path(os.path.join(_hermes_home, "profiles", p, "config.yaml"))
        if not config.exists():
            print(f"  ? {p} config 不存在")
            continue
        injected, why = _profile_injected(config)
        print(f"  {'✓' if injected else '✗'} {p}"
              + ("" if injected else f"（{why}）"))
        all_injected = all_injected and injected
    if not all_injected:
        return False

    # [8] Gateway（权威判据）
    print("\n[8] Gateway 进程")
    active, why = _gateway_active()
    print(f"  {'✓' if active else '✗'} {why}")
    if not active:
        return False

    print("\n" + "=" * 60)
    print("[OK] 健康检查通过")
    print("=" * 60)
    return True


if __name__ == "__main__":
    ok = check()
    sys.exit(0 if ok else 1)
