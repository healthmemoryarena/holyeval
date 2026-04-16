# Appium Integration — DeviceDriver 抽象层

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 GUI Agent 的设备操作抽象为 DeviceDriver 接口，保留 ADB 直连并新增 Appium 实现，通过 `--driver` 参数切换。

**Architecture:** 从 `test_gui_agent.py` 中提取设备操作到 `drivers/` 包（ABC + ADB/Appium 实现），同时提取 VLM 后端到 `backends/` 包。主流程通过工厂函数获取 driver 实例，VLM 后端和设备操作完全解耦。

**Tech Stack:** Python 3.11+, Appium-Python-Client (可选依赖), adb

---

## Task 1: 创建 DeviceDriver 抽象基类

**Files:**
- Create: `scripts/gui_agent/drivers/__init__.py`
- Create: `scripts/gui_agent/drivers/base.py`

**Step 1: 创建 drivers 包和 ABC**

`scripts/gui_agent/drivers/base.py`:
```python
from abc import ABC, abstractmethod


class DeviceDriver(ABC):
    """设备操作抽象基类，所有坐标为实际像素值"""

    @abstractmethod
    def get_screen_size(self) -> tuple[int, int]:
        """返回 (width, height) 像素"""

    @abstractmethod
    def screenshot(self) -> str:
        """截图并返回 base64 编码的 PNG"""

    @abstractmethod
    def tap(self, x: int, y: int) -> None: ...

    @abstractmethod
    def long_press(self, x: int, y: int, duration_ms: int = 1500) -> None: ...

    @abstractmethod
    def input_text(self, text: str, x: int | None = None, y: int | None = None) -> None:
        """输入文字。若提供坐标，先点击激活输入框。"""

    @abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None: ...

    @abstractmethod
    def press_key(self, key: str) -> None:
        """key: 'home' | 'back' | 'enter'"""

    @abstractmethod
    def launch_app(self, package_or_name: str) -> None: ...

    def close(self) -> None:
        """清理资源，默认无操作"""
        pass
```

`scripts/gui_agent/drivers/__init__.py`:
```python
from .base import DeviceDriver

__all__ = ["DeviceDriver"]
```

**Step 2: 验证 import 无报错**

Run: `cd /Users/admin/workspace/holyeval && python -c "from scripts.gui_agent.drivers import DeviceDriver; print('OK')"`

如果模块路径不通（scripts 不是 package），改用:
```bash
cd /Users/admin/workspace/holyeval/scripts/gui_agent && python -c "from drivers import DeviceDriver; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/gui_agent/drivers/
git commit -m "feat(gui-agent): add DeviceDriver abstract base class"
```

---

## Task 2: 迁移 ADB 逻辑到 AdbDriver

**Files:**
- Create: `scripts/gui_agent/drivers/adb_driver.py`
- Modify: `scripts/gui_agent/drivers/__init__.py`

**Step 1: 实现 AdbDriver**

从 `test_gui_agent.py` 的 `adb_cmd()`, `get_screen_size()`, `take_screenshot()`, `execute_action()` 中提取设备操作逻辑。

`scripts/gui_agent/drivers/adb_driver.py`:
```python
import base64
import re
import subprocess
import time

from .base import DeviceDriver

PACKAGE_MAP = {
    "theta": "com.thetaai.theta",
    "theta health": "com.thetaai.theta",
    "设置": "com.android.settings",
    "settings": "com.android.settings",
    "微信": "com.tencent.mm",
    "浏览器": "com.android.browser",
}


class AdbDriver(DeviceDriver):
    """通过 ADB 命令直连 Android 设备"""

    def _cmd(self, cmd: str) -> str:
        result = subprocess.run(
            f"adb {cmd}", shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()

    def get_screen_size(self) -> tuple[int, int]:
        output = self._cmd("shell wm size")
        match = re.search(r"(\d+)x(\d+)", output)
        if match:
            return int(match.group(1)), int(match.group(2))
        raise RuntimeError(f"无法获取屏幕尺寸: {output}")

    def screenshot(self) -> str:
        import tempfile, os
        remote_path = f"/sdcard/gui_agent_screenshot_{int(time.time())}.png"
        local_path = os.path.join(tempfile.gettempdir(), f"gui_screenshot_{int(time.time())}.png")
        self._cmd(f"shell screencap -p {remote_path}")
        self._cmd(f"pull {remote_path} {local_path}")
        self._cmd(f"shell rm {remote_path}")
        with open(local_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(local_path)
        return b64

    def tap(self, x: int, y: int) -> None:
        self._cmd(f"shell input tap {x} {y}")

    def long_press(self, x: int, y: int, duration_ms: int = 1500) -> None:
        self._cmd(f"shell input swipe {x} {y} {x} {y} {duration_ms}")

    def input_text(self, text: str, x: int | None = None, y: int | None = None) -> None:
        if x is not None and y is not None:
            self._cmd(f"shell input tap {x} {y}")
            time.sleep(0.5)
        self._cmd(
            f'shell app_process -Djava.class.path=/data/local/tmp/yadb '
            f'/data/local/tmp com.ysbing.yadb.Main -keyboard "{text}"'
        )

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        self._cmd(f"shell input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def press_key(self, key: str) -> None:
        key_map = {"home": 3, "back": 4, "enter": 66}
        code = key_map.get(key)
        if code is None:
            raise ValueError(f"未知按键: {key}，支持: {list(key_map.keys())}")
        self._cmd(f"shell input keyevent {code}")

    def launch_app(self, package_or_name: str) -> None:
        pkg = PACKAGE_MAP.get(package_or_name.lower(), package_or_name)
        self._cmd(f"shell monkey -p {pkg} -c android.intent.category.LAUNCHER 1")
```

**Step 2: 更新 `__init__.py`**

```python
from .base import DeviceDriver
from .adb_driver import AdbDriver

__all__ = ["DeviceDriver", "AdbDriver"]
```

**Step 3: 验证 import**

Run: `cd /Users/admin/workspace/holyeval/scripts/gui_agent && python -c "from drivers import AdbDriver; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add scripts/gui_agent/drivers/
git commit -m "feat(gui-agent): implement AdbDriver from existing ADB logic"
```

---

## Task 3: 实现 AppiumDriver

**Files:**
- Create: `scripts/gui_agent/drivers/appium_driver.py`
- Modify: `scripts/gui_agent/drivers/__init__.py`

**Step 1: 实现 AppiumDriver**

`scripts/gui_agent/drivers/appium_driver.py`:
```python
from .base import DeviceDriver

PACKAGE_MAP = {
    "theta": "com.thetaai.theta",
    "theta health": "com.thetaai.theta",
    "设置": "com.android.settings",
    "settings": "com.android.settings",
    "微信": "com.tencent.mm",
    "浏览器": "com.android.browser",
}


def _import_appium():
    """动态导入 Appium，缺失时给友好提示"""
    try:
        from appium import webdriver
        from appium.options.android import UiAutomator2Options
        return webdriver, UiAutomator2Options
    except ImportError:
        raise ImportError(
            "Appium 依赖未安装。请执行:\n"
            "  pip install Appium-Python-Client\n"
            "  npm install -g appium\n"
            "  appium driver install uiautomator2"
        )


class AppiumDriver(DeviceDriver):
    """通过 Appium Server 控制 Android 设备"""

    def __init__(self, appium_url: str = "http://localhost:4723", device_name: str = ""):
        webdriver, UiAutomator2Options = _import_appium()
        options = UiAutomator2Options()
        options.no_reset = True
        options.auto_grant_permissions = True
        if device_name:
            options.device_name = device_name
        self._driver = webdriver.Remote(appium_url, options=options)
        self._screen_size: tuple[int, int] | None = None

    def get_screen_size(self) -> tuple[int, int]:
        if self._screen_size is None:
            size = self._driver.get_window_size()
            self._screen_size = (size["width"], size["height"])
        return self._screen_size

    def screenshot(self) -> str:
        return self._driver.get_screenshot_as_base64()

    def tap(self, x: int, y: int) -> None:
        from appium.webdriver.common.touch_action import TouchAction
        TouchAction(self._driver).tap(x=x, y=y).perform()

    def long_press(self, x: int, y: int, duration_ms: int = 1500) -> None:
        from appium.webdriver.common.touch_action import TouchAction
        TouchAction(self._driver).long_press(x=x, y=y, duration=duration_ms).perform()

    def input_text(self, text: str, x: int | None = None, y: int | None = None) -> None:
        if x is not None and y is not None:
            self.tap(x, y)
            import time
            time.sleep(0.5)
        # Appium 的 send_keys 需要一个 active element
        el = self._driver.switch_to.active_element
        el.send_keys(text)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        self._driver.swipe(x1, y1, x2, y2, duration_ms)

    def press_key(self, key: str) -> None:
        key_map = {"home": 3, "back": 4, "enter": 66}
        code = key_map.get(key)
        if code is None:
            raise ValueError(f"未知按键: {key}，支持: {list(key_map.keys())}")
        self._driver.press_keycode(code)

    def launch_app(self, package_or_name: str) -> None:
        pkg = PACKAGE_MAP.get(package_or_name.lower(), package_or_name)
        self._driver.activate_app(pkg)

    def close(self) -> None:
        try:
            self._driver.quit()
        except Exception:
            pass
```

**Step 2: 更新 `__init__.py` 加入工厂函数**

```python
from .base import DeviceDriver
from .adb_driver import AdbDriver

__all__ = ["DeviceDriver", "AdbDriver", "create_driver"]


def create_driver(driver_type: str = "adb", **kwargs) -> DeviceDriver:
    """工厂函数：根据类型创建 driver 实例"""
    if driver_type == "adb":
        return AdbDriver()
    elif driver_type == "appium":
        from .appium_driver import AppiumDriver  # 延迟导入
        return AppiumDriver(**kwargs)
    else:
        raise ValueError(f"未知 driver 类型: {driver_type}，支持: adb, appium")
```

**Step 3: 验证 import（不需要 Appium 安装）**

Run: `cd /Users/admin/workspace/holyeval/scripts/gui_agent && python -c "from drivers import create_driver; d = create_driver('adb'); print('OK')"`
Expected: `OK`（ADB driver 不需要额外依赖）

**Step 4: Commit**

```bash
git add scripts/gui_agent/drivers/
git commit -m "feat(gui-agent): implement AppiumDriver for Android"
```

---

## Task 4: 提取 VLM 后端到 backends/ 包

**Files:**
- Create: `scripts/gui_agent/backends/__init__.py`
- Create: `scripts/gui_agent/backends/base.py`
- Create: `scripts/gui_agent/backends/ui_tars.py`
- Create: `scripts/gui_agent/backends/gelab_zero.py`

**Step 1: 提取代码**

从 `test_gui_agent.py` 中剪切 `AgentBackend` ABC → `backends/base.py`，`UITarsBackend` → `backends/ui_tars.py`，`GElabZeroBackend` → `backends/gelab_zero.py`。

`backends/__init__.py`:
```python
from .base import AgentBackend
from .ui_tars import UITarsBackend
from .gelab_zero import GElabZeroBackend

BACKENDS: dict[str, type[AgentBackend]] = {
    "ui-tars": UITarsBackend,
    "gelab-zero": GElabZeroBackend,
}

__all__ = ["AgentBackend", "BACKENDS"]
```

每个文件只搬代码，不改逻辑。`base.py` 含 ABC + `re` import，`ui_tars.py` 和 `gelab_zero.py` 各含对应类 + `import re`。

**Step 2: 验证 import**

Run: `cd /Users/admin/workspace/holyeval/scripts/gui_agent && python -c "from backends import BACKENDS; print(list(BACKENDS.keys()))"`
Expected: `['ui-tars', 'gelab-zero']`

**Step 3: Commit**

```bash
git add scripts/gui_agent/backends/
git commit -m "refactor(gui-agent): extract VLM backends to backends/ package"
```

---

## Task 5: 改造主流程 test_gui_agent.py

**Files:**
- Modify: `scripts/gui_agent/test_gui_agent.py`

**Step 1: 重写主流程**

删除已迁移到 drivers/ 和 backends/ 的代码（`adb_cmd`, `get_screen_size`, `take_screenshot`, `execute_action`, `AgentBackend`, `UITarsBackend`, `GElabZeroBackend`, `BACKENDS`, `PACKAGE_MAP`）。

替换为:
- `from drivers import create_driver`
- `from backends import BACKENDS`
- 新的 `execute_action(action, driver, screen_w, screen_h)` 使用 driver 接口
- CLI 加 `--driver adb|appium` 和 `--appium-url` 参数
- 截图保存逻辑保留在主流程（driver.screenshot() 只返回 base64）

关键改动 — `execute_action`:
```python
def execute_action(action: dict, driver: DeviceDriver, screen_w: int, screen_h: int) -> bool:
    """执行动作，返回 False 表示任务结束"""
    t = action["action_type"]
    p = action["params"]

    if t == "click":
        x = int(p["x"] / 1000 * screen_w)
        y = int(p["y"] / 1000 * screen_h)
        driver.tap(x, y)
        print(f"  -> tap ({x}, {y})")

    elif t == "long_press":
        x = int(p["x"] / 1000 * screen_w)
        y = int(p["y"] / 1000 * screen_h)
        driver.long_press(x, y)
        print(f"  -> long_press ({x}, {y})")

    elif t == "type":
        content = p.get("content", "")
        tx = int(p["x"] / 1000 * screen_w) if "x" in p else None
        ty = int(p["y"] / 1000 * screen_h) if "y" in p else None
        driver.input_text(content, tx, ty)
        print(f"  -> type '{content}'")

    elif t == "scroll":
        x = int(p.get("x", 500) / 1000 * screen_w)
        y = int(p.get("y", 500) / 1000 * screen_h)
        direction = p.get("direction", "down")
        delta = int(screen_h * 0.3)
        swipe_map = {
            "down": (x, y, x, y - delta),
            "up": (x, y, x, y + delta),
            "left": (x, y, x + delta, y),
            "right": (x, y, x - delta, y),
        }
        x1, y1, x2, y2 = swipe_map.get(direction, (x, y, x, y - delta))
        driver.swipe(x1, y1, x2, y2)
        print(f"  -> scroll {direction} at ({x}, {y})")

    elif t == "open_app":
        app_name = p.get("app_name", "")
        driver.launch_app(app_name)
        print(f"  -> open_app '{app_name}'")

    elif t == "press_home":
        driver.press_key("home")
        print("  -> press_home")

    elif t == "press_back":
        driver.press_key("back")
        print("  -> press_back")

    elif t == "press_enter":
        driver.press_key("enter")
        print("  -> press_enter")

    elif t == "finished":
        reason = p.get("content", "done")
        print(f"  -> finished: {reason}")
        return False

    elif t == "drag":
        x1 = int(p["x"] / 1000 * screen_w)
        y1 = int(p["y"] / 1000 * screen_h)
        x2 = int(p["end_x"] / 1000 * screen_w)
        y2 = int(p["end_y"] / 1000 * screen_h)
        driver.swipe(x1, y1, x2, y2)
        print(f"  -> drag ({x1},{y1}) -> ({x2},{y2})")

    elif t == "wait":
        import time
        duration = int(p.get("duration", 3))
        print(f"  -> wait {duration}s")
        time.sleep(duration)

    else:
        print(f"  ! 未知动作: {t}")

    return True
```

CLI 新增参数:
```python
parser.add_argument("--driver", choices=["adb", "appium"], default="adb", help="设备驱动 (默认: adb)")
parser.add_argument("--appium-url", default="http://localhost:4723", help="Appium Server 地址")
```

main() 中创建 driver:
```python
driver = create_driver(args.driver, appium_url=args.appium_url)
screen_w, screen_h = driver.get_screen_size()
# ... 主循环中用 driver.screenshot() 替代 take_screenshot()
# ... finally: driver.close()
```

**Step 2: 验证语法**

Run: `cd /Users/admin/workspace/holyeval && python -m py_compile scripts/gui_agent/test_gui_agent.py && echo "OK"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/gui_agent/test_gui_agent.py
git commit -m "refactor(gui-agent): use DeviceDriver interface, add --driver flag"
```

---

## Task 6: 端到端验证（ADB driver）

**Step 1: 用 ADB driver 跑一次**

```bash
cd /Users/admin/workspace/holyeval
python scripts/gui_agent/test_gui_agent.py --agent ui-tars --driver adb "打开设置"
```

验证行为和重构前一致。

**Step 2: 验证 Appium 缺失时的报错**

```bash
python scripts/gui_agent/test_gui_agent.py --driver appium "打开设置"
```

Expected: 友好错误提示，不是 traceback。

**Step 3: Commit（如有修复）**

```bash
git add -A && git commit -m "fix(gui-agent): post-refactor fixes"
```
