"""从镜像构建阶段移除安装工具、测试数据和字节码。"""
from pathlib import Path
import shutil
import sys
import sysconfig


def main():
    root = Path(sysconfig.get_path("stdlib")).resolve()
    if sys.platform != "linux" or root != Path("/usr/local/lib/python3.12") or "--apply" not in sys.argv:
        raise SystemExit("Run with --apply only inside the Python 3.12 image build stage")
    site = root / "site-packages"
    targets = [root / "ensurepip", site / "pip", *site.glob("pip-*.dist-info")]
    # 保留 numpy.testing 辅助功能、运行库和许可证，仅移除
    # NumPy 自带的测试集与生成的字节码。
    targets.extend((site / "numpy").rglob("tests"))
    targets.extend(root.rglob("__pycache__"))
    removed_bytes = 0
    for target in sorted(set(targets), key=lambda path: len(path.parts), reverse=True):
        if not target.exists():
            continue
        if not target.resolve().is_relative_to(root):
            raise RuntimeError("Cleanup target is outside the Python runtime: " + str(target))
        removed_bytes += sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
        shutil.rmtree(target)
    for name in ("pip", "pip3", "pip3.12"):
        (Path("/usr/local/bin") / name).unlink(missing_ok=True)
    print(f"Removed {removed_bytes} bytes of installation tools, bundled tests and bytecode")


if __name__ == "__main__":
    main()
