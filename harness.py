"""将赛事宿主命令交给 contest_harness SDK，执行协议声明、训练或预测。"""
import sys
from contest_harness.cli import main
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

