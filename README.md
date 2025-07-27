# ZJU-DEM-su25-DynamicPricing

## 项目结构

- `docs`: 本项目的文档

- `src`: 项目源码

## 运行方法

- 确保你的电脑上安装了 python 3.12 环境和 pip

- 进入项目根目录，执行 `pip install -r requirements.txt` 安装依赖

- 运行 `python src/main.py` 启动程序，运行参数：

```
python3 main.py --help
Usage: main.py [OPTIONS]

Options:
  -n INTEGER  Number of data
  -m INTEGER  Type of buyers
  -t INTEGER  Turns of the game
  -s INTEGER  Type of seller curve (0: smooth, 1: diminishing)
  -p INTEGER  Method of buyer generation (0: random, 1: adversarial)
  --help      Show this message and exit.
```
