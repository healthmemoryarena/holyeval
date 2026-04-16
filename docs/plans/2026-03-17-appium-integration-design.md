# Appium 集成设计 — GUI Agent DeviceDriver 抽象层

## 背景

当前 `scripts/gui_agent/test_gui_agent.py` 通过 ADB 直连 Android 真机执行 GUI 测试。为支持未来 iOS 和模拟器测试，需引入 Appium 作为统一自动化层，同时保留 ADB 直连作为备选。

## 设计决策

- **方案 A（采用）**：抽象类 + 多文件拆分，`DeviceDriver` ABC + 多实现
- 保留 ADB 直连（`--driver adb`，默认）
- 先做 Android Appium，iOS 留接口
- 独立脚本迭代，不集成到 HolyEval TargetAgent 插件

## 文件结构

```
scripts/gui_agent/
├── test_gui_agent.py              # 主流程（改用 driver 接口）
├── drivers/
│   ├── __init__.py                # create_driver() 工厂函数
│   ├── base.py                    # DeviceDriver ABC
│   ├── adb_driver.py              # 现有 ADB 逻辑迁移
│   └── appium_driver.py           # Appium Android 实现
├── backends/
│   ├── __init__.py                # BACKENDS 注册表
│   ├── base.py                    # AgentBackend ABC
│   ├── ui_tars.py                 # UI-TARS-72B
│   └── gelab_zero.py              # GELab-Zero-4B
├── runs/                          # 运行日志
└── start_gelab_vllm.sh
```

## DeviceDriver 接口

```python
class DeviceDriver(ABC):
    def get_screen_size(self) -> tuple[int, int]
    def screenshot(self) -> str              # 返回 base64
    def tap(self, x: int, y: int)
    def long_press(self, x: int, y: int)
    def input_text(self, text: str, x: int | None, y: int | None)
    def swipe(self, x1, y1, x2, y2, duration_ms=500)
    def press_key(self, key: str)            # "home" | "back" | "enter"
    def launch_app(self, package_or_name: str)
    def close(self)                          # 清理资源
```

- 坐标归一化（0-1000 → 像素）在 execute_action 中完成，driver 接收实际像素
- AdbDriver: 包装现有 subprocess + adb 命令
- AppiumDriver: 通过 Appium-Python-Client 连接 Appium Server

## CLI

```bash
# 默认 ADB（向后兼容）
python scripts/gui_agent/test_gui_agent.py --agent ui-tars "打开theta"

# Appium
python scripts/gui_agent/test_gui_agent.py --driver appium "打开theta"
python scripts/gui_agent/test_gui_agent.py --driver appium --appium-url http://localhost:4723 "打开theta"
```

## 依赖

Appium 作为可选依赖，不加到 pyproject.toml：
- `pip install Appium-Python-Client`
- `npm install -g appium && appium driver install uiautomator2`
- `--driver appium` 时动态 import，缺失时报友好错误

## 未来扩展

- iOS: 新增 `appium_ios_driver.py`，安装 `appium driver install xcuitest`
- XML dump: 在 DeviceDriver 加 `get_page_source() -> str` 方法
- 集成到 HolyEval: 包装为 `gui_app` TargetAgent 插件
