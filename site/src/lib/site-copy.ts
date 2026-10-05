import { useI18n } from "fumadocs-ui/contexts/i18n";

import { DEFAULT_LOCALE, isLocale, type Locale } from "@/lib/i18n";

const zhCN = {
  pageRailLabel: "页面区块导航",
  theme: {
    hint: "点击切换主题",
    modes: { system: "主题：跟随系统", light: "主题：浅色", dark: "主题：深色" },
  },
  home: {
    sections: ["首页", "功能", "使用流程", "特点", "安装"],
    hero: {
      eyebrow: "基于检索的语音转换（RVC），一体化重写",
      title: "RVC 的全部工作，",
      titleAccent: "收进同一个应用",
      description:
        "RVC Next 用一个服务端、一个网页界面和一套命令行完成音色转换、实时变声、人声分离、模型训练与模型管理；数值计算从原版逐行移植，输出与原版一致。",
      start: "开始使用",
      learn: "了解",
      platformsLabel: "支持的平台",
    },
    features: {
      title: "五个界面，覆盖完整的变声流程",
      description: "每个界面都有对应的教程；同样的操作也都可以在命令行中完成。",
      open: "查看教程",
      items: [
        {
          title: "转换",
          description: "用音色转换单个文件或整个文件夹，可先分离人声、再把伴奏混回去。",
          href: "/docs/guide-convert",
        },
        {
          title: "实时变声",
          description: "用服务端电脑的声卡实时变声；音色、音高与缓冲参数都能在运行中切换。",
          href: "/docs/guide-live",
        },
        {
          title: "人声分离",
          description: "分离人声与伴奏、去混响、提取主唱，也可以把多个步骤串成一个预设。",
          href: "/docs/guide-separate",
        },
        {
          title: "训练",
          description: "从录音到音色：切片、音高、特征、训练、索引，改动后只重跑需要的步骤。",
          href: "/docs/guide-train",
        },
        {
          title: "模型管理",
          description: "按内容识别并导入音色与索引，下载示例音色、底模、分离模型和所需资源。",
          href: "/docs/guide-models",
        },
      ],
    },
    workflow: {
      title: "从安装到第一个属于你的音色",
      description: "四步走完整个流程，每一步在网页界面和命令行中都有对应做法。",
      steps: [
        {
          title: "安装",
          description: "在独立的 Python 环境中先装 PyTorch，再用 pip 安装 RVC Next。",
        },
        {
          title: "下载模型",
          description: "首次使用时一键下载 HuBERT 与音高模型，再挑一个示例音色。",
        },
        {
          title: "转换与实时变声",
          description: "选择音色、调整音高，转换音频文件，或接上麦克风实时变声。",
        },
        {
          title: "训练自己的音色",
          description: "准备录音，运行训练，试听检查点，满意后导出到音色库。",
        },
      ],
    },
    highlights: {
      title: "与原版一致，用起来更省心",
      description: "保留原版的数值结果与模型格式，把重复劳动交给程序处理。",
      items: [
        {
          title: "输出与原版逐位一致",
          description: "离线转换与实时变声都用对比测试与原版核对，现有模型和索引直接可用。",
        },
        {
          title: "一个服务端，网页与命令行同源",
          description: "网页界面与命令行调用同一套核心操作，命令行的 JSON 输出与 API 完全相同。",
        },
        {
          title: "实时变声可热切换",
          description: "运行中切换音色、音高算法和参数，新音色在两个音频块之间无缝换上。",
        },
        {
          title: "导入按内容识别",
          description: "无论文件名如何，都能认出音色、索引、底模与分离模型，并自动配对。",
        },
        {
          title: "训练只重跑改动部分",
          description: "每个阶段记录输入指纹，数据或设置变化时只重做受影响的步骤。",
        },
      ],
    },
    install: {
      title: "几条命令即可开始",
      description: "先在独立环境中安装适合硬件的 PyTorch，再安装 RVC Next 并启动网页界面。",
      link: "阅读安装文档",
    },
    footer: {
      label: "页脚导航",
      docs: "文档",
      quickStart: "快速开始",
      faq: "常见问题",
      repository: "源代码",
      socialLabel: "社交链接",
      social: { blog: "博客", github: "GitHub", bilibili: "哔哩哔哩" },
    },
    preview: {
      label: "RVC Next 网页界面交互示意",
      pause: "暂停演示",
      play: "播放演示",
      workspace: "音色库",
      select: "选择一个音色",
      create: "导入模型",
      search: "搜索音色",
      voices: [
        {
          detail: "v1 · 40 kHz · 有索引",
          status: "转换中",
          operationTitle: "转换 song.wav",
          operationDetail: "RMVPE · 音高 +12",
        },
        {
          detail: "v2 · 40 kHz · 有索引",
          status: "实时变声",
          operationTitle: "实时变声运行中",
          operationDetail: "48 kHz · 双工",
        },
        {
          detail: "v2 · 48 kHz · 训练中",
          status: "训练中",
          operationTitle: "训练 · 第 12/20 轮",
          operationDetail: "检查点已保存",
        },
      ],
    },
  },
} as const;

type TranslationShape<T> = T extends string
  ? string
  : T extends readonly unknown[]
    ? { readonly [K in keyof T]: TranslationShape<T[K]> }
    : T extends object
      ? { readonly [K in keyof T]: TranslationShape<T[K]> }
      : T;

export type SiteCopy = TranslationShape<typeof zhCN>;

const en = {
  pageRailLabel: "Page sections",
  theme: {
    hint: "Click to switch theme",
    modes: { system: "Theme: follow system", light: "Theme: light", dark: "Theme: dark" },
  },
  home: {
    sections: ["Home", "Features", "Workflow", "Highlights", "Install"],
    hero: {
      eyebrow: "Retrieval-based Voice Conversion (RVC), rewritten as one app",
      title: "Everything RVC does,",
      titleAccent: "in one application",
      description:
        "RVC Next converts voices, changes your voice live, separates vocals, trains models and manages them from one server, one web UI and one command line. Its numerical code is ported line by line from the original, with the same output.",
      start: "Get started",
      learn: "Explore ",
      platformsLabel: "Supported platforms",
    },
    features: {
      title: "Five screens for the whole voice conversion workflow",
      description:
        "Every screen has its own tutorial, and everything it does can also be done from the command line.",
      open: "Read the tutorial",
      items: [
        {
          title: "Convert",
          description:
            "Convert a file or a whole folder with a voice; separate the vocals first and mix the accompaniment back in.",
          href: "/docs/guide-convert",
        },
        {
          title: "Live",
          description:
            "Change your voice live through the server's audio devices; switch voice, pitch and buffering while it runs.",
          href: "/docs/guide-live",
        },
        {
          title: "Separate",
          description:
            "Split vocals from accompaniment, remove reverb, extract the lead vocal, or chain several steps into one preset.",
          href: "/docs/guide-separate",
        },
        {
          title: "Train",
          description:
            "From recordings to a voice: slicing, pitch, features, training and index, rerunning only what changed.",
          href: "/docs/guide-train",
        },
        {
          title: "Models",
          description:
            "Import voices and indexes identified by their content, and download demo voices, base models, separation models and assets.",
          href: "/docs/guide-models",
        },
      ],
    },
    workflow: {
      title: "From installation to a voice of your own",
      description:
        "Four steps through the whole workflow, each with a web UI and a command-line way to do it.",
      steps: [
        {
          title: "Install",
          description:
            "In its own Python environment, install PyTorch first, then RVC Next with pip.",
        },
        {
          title: "Download the models",
          description: "Download HuBERT and the pitch models in one click, then pick a demo voice.",
        },
        {
          title: "Convert and go live",
          description:
            "Choose a voice, set the pitch, convert audio files or change your voice live from a microphone.",
        },
        {
          title: "Train your own voice",
          description:
            "Prepare recordings, run the training, try the checkpoints and export the one you like to the library.",
        },
      ],
    },
    highlights: {
      title: "Faithful to the original, easier to use",
      description:
        "The original's numerical results and model formats stay; the repetitive work moves to the program.",
      items: [
        {
          title: "Bit-identical output",
          description:
            "Offline conversion and live voice changing are checked against the original by comparison tests; existing models and indexes work as they are.",
        },
        {
          title: "One server for the web UI and the command line",
          description:
            "Both call the same core operations, and the command line's JSON output equals the API's.",
        },
        {
          title: "Hot swaps in live voice changing",
          description:
            "Switch voice, pitch method and parameters while running; a new voice comes in between two audio blocks.",
        },
        {
          title: "Imports identified by content",
          description:
            "Voices, indexes, base models and separation models are recognised and paired whatever their file names.",
        },
        {
          title: "Training reruns only what changed",
          description:
            "Every stage records a fingerprint of its inputs, so a change redoes only the steps it affects.",
        },
      ],
    },
    install: {
      title: "A few commands to get started",
      description:
        "Install PyTorch for your hardware in its own environment, then install RVC Next and start the web UI.",
      link: "Read the installation guide",
    },
    footer: {
      label: "Footer navigation",
      docs: "Docs",
      quickStart: "Quick start",
      faq: "FAQ",
      repository: "Source code",
      socialLabel: "Social links",
      social: { blog: "Blog", github: "GitHub", bilibili: "Bilibili" },
    },
    preview: {
      label: "Interactive sketch of the RVC Next web UI",
      pause: "Pause the demo",
      play: "Play the demo",
      workspace: "Voice library",
      select: "Choose a voice",
      create: "Import models",
      search: "Search voices",
      voices: [
        {
          detail: "v1 · 40 kHz · indexed",
          status: "Converting",
          operationTitle: "Converting song.wav",
          operationDetail: "RMVPE · pitch +12",
        },
        {
          detail: "v2 · 40 kHz · indexed",
          status: "Live",
          operationTitle: "Live voice changing",
          operationDetail: "48 kHz · duplex",
        },
        {
          detail: "v2 · 48 kHz · training",
          status: "Training",
          operationTitle: "Training · epoch 12/20",
          operationDetail: "Checkpoint saved",
        },
      ],
    },
  },
} as const satisfies SiteCopy;

const SITE_COPY: Record<Locale, SiteCopy> = { "zh-CN": zhCN, en };

export function getSiteCopy(locale: Locale): SiteCopy {
  return SITE_COPY[locale];
}

export function useSiteLocale(): Locale {
  const { locale } = useI18n();
  return isLocale(locale) ? locale : DEFAULT_LOCALE;
}

export function useSiteCopy(): SiteCopy {
  return getSiteCopy(useSiteLocale());
}
