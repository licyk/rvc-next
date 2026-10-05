# rvc-next

[English](README.md) | 简体中文

基于检索的语音转换（RVC），重写为一个完整的应用：**转换**音频文件、通过声卡**实时**变声、从音乐中**分离**人声、**训练**自己的音色，并在同一个**模型库**中管理它们，网页界面和命令行都能完成。

rvc-next 是 [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的重写。数值计算部分（合成器、RMVPE、转换与实时流的计算、训练循环）从原版移植，因此已有的音色听起来完全一样；除此之外的一切都是新的：

- **到处都是同一个音色选择器、同一个参数面板**。每个音色有自己的默认预设，在转换和实时变声中听起来一致。
- **每个耗时操作都是一个任务**，可以查看进度、取消、重试，稍后再回来查看；界面不会被卡住。
- **流水线**：在一个任务里完成分离人声、转换、再混回伴奏。
- **音频设备**按物理设备列出，每个设备只出现一次；输入和输出可以使用任意驱动，可选监听输出，带电平表和测试音，设备重新接入后自动恢复；还可以用回环测试实测真实延迟，而不只是估算。
- **已有模型无需改动即可加载**（`.pth` 和 `added_*.index`），可以直接读取已有的 RVC 安装，新生成的模型也使用同样的格式。

文档（使用教程、命令行与开发文档，中英双语）位于 [site/content/docs/](site/content/docs/index.mdx)，是 `site/` 中项目网站的一部分（运行 `python scripts/dev.py site-dev` 可本地预览）。设计记录、约定和已知的不足见 [AGENTS.md](AGENTS.md)。

## 安装

请使用全新的虚拟环境（或独立的 Python）：rvc-next 把依赖固定在测试过的版本范围内，不适合与其他应用共用环境。先按硬件安装 PyTorch（2.7.1 或更新；不需要 torchaudio）：

| 硬件 | PyTorch 构建（`--index-url https://download.pytorch.org/whl/…`） |
| --- | --- |
| NVIDIA | `cu128`（RTX 50 系列必须用它；较早的显卡也可以用其他 CUDA 构建） |
| AMD（Linux） | `rocm7.2`；显卡显示为 `cuda:N` |
| Intel Arc 及较新的核显 | `xpu`（Linux 和 Windows）；显卡显示为 `xpu:N` |
| Apple 芯片 | 默认构建（`pip install torch`）；Metal 支持为实验性 |
| 仅 CPU | `cpu` |

在 Windows 上，AMD 和 Intel 显卡也可以使用 DirectML：安装 CPU 版 PyTorch，再运行 `pip install "rvc-next[directml]"`。

以 NVIDIA 显卡为例：

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install rvc-next
rvc-next assets download          # HuBERT 及音高模型 RMVPE、FCPE（约 400 MB）
rvc-next model download --all     # 可选：RVC 官方演示音色
rvc-next webui                    # http://127.0.0.1:7868
```

`rvc-next doctor` 会报告 torch 的构建版本（CUDA、ROCm 或 XPU）、GPU 规则选中的设备、音频设备访问情况以及资源状态。ROCm、XPU、DirectML 和 Metal 尚未在真实硬件上验证。在 Linux 上，实时变声需要 PortAudio（`libportaudio2`）。

如果要复用已有 RVC 安装中的模型，把 `paths.assets_dir` 指向它的 `assets/` 文件夹，并把该安装添加为原版 RVC 安装（**模型** › **音色** › **原版 RVC 安装**；或运行 `rvc-next model import <RVC 文件夹>`，把其中的音色复制到模型库）。

## 模型

模型从 [licyk/rvc-model](https://huggingface.co/licyk/rvc-model) 下载，这个仓库按类别存放了 rvc-next 用到的全部模型，以及 RVC 官方演示音色。也可以改用官方仓库 [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)（运行 `rvc-next config set downloads.repository official`，或在 **设置** › **下载** 中切换）；`downloads.source` 可以把下载地址切换到 `hf-mirror.com` 或自定义地址。

自己的模型可以在 **模型** 界面导入，或用 `rvc-next model import` 导入：音色（`.pth`，连同其 `.index`）、训练用的底模（G 和 D 的 `.pth`）以及分离模型（检查点连同其 `.yaml`），可以是散装文件，也可以打包成 `.zip`。文件按内容识别，所以即使 `.pth` 和 `.index` 的文件名毫不相关，也能正确配对：索引的维度决定它适用的音色版本；有多个音色都符合时，由你选择。索引也可以单独导入，之后再分配给某个音色。导入的底模会出现在实验设置中，导入的分离模型会作为预设出现在 **分离** 界面。

## 命令行

```bash
rvc-next model import voice.zip --name "My voice"
rvc-next convert song.wav --voice "My voice" --pitch 2 --separate vocals --remix -o out/
rvc-next separate track.flac --preset vocals-clean
rvc-next train new alice --dataset ~/datasets/alice && rvc-next train run alice
rvc-next live devices && rvc-next live run --voice "My voice"
rvc-next --help
```

所有列表命令都支持 `--json`，输出的结构与 API 返回的相同。

## 数据

设置、数据库、音色库、实验和输出文件存放在 `$XDG_DATA_HOME/rvc-next`、`~/Library/Application Support/rvc-next` 或 `%APPDATA%\rvc-next`（可用 `RVC_NEXT_DATA_DIR` 更改位置）。任何设置都可以用环境变量 `RVC_NEXT_<分组>__<字段>` 覆盖，例如 `RVC_NEXT_COMPUTE__DEVICE=cpu`。

## 许可证

GPL-3.0。从 RVC（MIT）及其他项目移植的代码列在 [NOTICE](NOTICE) 中。
