<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <img src="https://github.com/facebookresearch/PurpleLlama/blob/main/logo.png" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/meta-Llama">Hugging Face 模型库</a>&nbsp | <a href="https://ai.meta.com/blog/purple-llama-open-trust-safety-generative-ai">官方博客</a>&nbsp |  <a href="https://ai.meta.com/llama/purple-llama">官方网站</a>&nbsp | <a href="https://ai.meta.com/research/publications/purple-llama-cyberseceval-a-benchmark-for-evaluating-the-cybersecurity-risks-of-large-language-models/">CyberSec Eval 论文</a>&nbsp&nbsp | <a href="https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/">Llama Guard 论文</a>&nbsp
<br>

---

# Purple Llama

Purple Llama 是一个全方位的综合性项目（Umbrella Project），旨在汇集各类安全工具与评测基准，协助开发者社区基于开源生成式 AI 模型进行安全、负责任的开发与构建。首发版本聚焦于**网络安全（Cyber Security）**以及**输入/输出安全防护机制（Input/Output Safeguards）**，后续我们将持续贡献更多安全工具与能力。

## 为什么命名为“紫色 (Purple)”？

借用网络安全领域的经典[概念](https://www.youtube.com/watch?v=ab_Fdp6FVDI)，我们认为要真正应对生成式 AI 带来的安全挑战，必须同时兼顾攻击视角（**红队 / Red Team**）与防御视角（**蓝队 / Blue Team**）。“**紫队（Purple Teaming）**”融合了红队与蓝队的双重视角与职责，是一种协作式评估与化解潜在安全风险的方法论。这一理念同样适用于生成式 AI 系统的安全建设，这也是我们在 Purple Llama 上进行全面投入的核心理念。

## 开源许可证 (License)

Purple Llama 项目内的组件采用宽松的开源许可证授权，支持学术研究与商业化应用。
我们坚信这是促进行业社区协作、推动生成式 AI 安全与可信工具标准化开发与应用的关键一步。
具体而言，各项评测工具与基准测试采用 MIT 许可证，而各类模型则采用对应的 Llama 社区许可证。详见下表：

| **组件类型** | **组件名称** | **开源许可证** |
| :--- | :---: | :---: |
| 评测基准 (Evals/Benchmarks) | Cyber Security Eval（更多基准持续推出中） | MIT |
| 安全护栏 (Safeguard) | Llama Guard | [Llama 2 Community License](https://github.com/facebookresearch/PurpleLlama/blob/main/LICENSE) |
| 安全护栏 (Safeguard) | Llama Guard 2 | [Llama 3 Community License](https://github.com/meta-llama/llama3/blob/main/LICENSE) |
| 安全护栏 (Safeguard) | Llama Guard 3-8B | [Llama 3.2 Community License](LICENSE) |
| 安全护栏 (Safeguard) | Llama Guard 3-1B | [Llama 3.2 Community License](LICENSE) |
| 安全护栏 (Safeguard) | Llama Guard 3-11B-vision | [Llama 3.2 Community License](LICENSE) |
| 安全护栏 (Safeguard) | Prompt Guard | [Llama 3.2 Community License](LICENSE) |
| 安全护栏 (Safeguard) | Code Shield | MIT |

## 系统级安全防护机制 (System-Level Safeguards)

正如我们在 Llama 3 的[负责任使用指南 (Responsible Use Guide)](https://ai.meta.com/llama/responsible-use-guide/) 中所指出的，我们建议针对大语言模型的所有输入（Input）与输出（Output），均结合具体应用场景的内容安全准则进行合规检测与过滤。

### Llama Guard

Llama Guard 3 包含一系列高性能的输入与输出内容审核模型，专为帮助开发者检测各类常见违规风险内容而设计。

该系列模型通过微调 Meta-Llama 3.1 和 3.2 模型构建，并针对 MLCommons 标准危害分类（Standard Hazards Taxonomy）进行了专门优化，可灵活适应各类开发者场景需求。
它们深度支持 Llama 3.2 的新特性，包括 7 种新增语言、128k 上下文窗口以及多模态图像推理能力。Llama Guard 3 模型还针对检测潜在的网络攻击辅助响应进行了深度优化，有效防止大语言模型生成的恶意代码在搭载代码解释器的 Llama 宿主环境中执行。

### Prompt Guard

Prompt Guard 是一款用于防护基于 LLM 应用免遭恶意 Prompt 攻击的强力工具，保障系统的安全性与完整性。

常见的提示词攻击主要包括**提示词注入（Prompt Injection）**与**越狱攻击（Jailbreaking）**：

* **提示词注入 (Prompt Injections)**：通过将来自第三方的不可信数据引入模型的上下文窗口，诱使模型执行违背开发者初衷的未授权指令。
* **越狱攻击 (Jailbreaks)**：旨在绕过或颠覆模型内置安全机制的对抗性恶意指令。

### Code Shield

Code Shield 支持在推理阶段对大语言模型生成的不安全代码进行实时过滤与拦截。Code Shield 能够有效降低不安全代码建议风险、防止代码解释器被恶意滥用，并确保安全执行命令。可参考 [CodeShield 使用示例 Notebook](https://github.com/meta-llama/PurpleLlama/blob/main/CodeShield/notebook/CodeShieldUsageDemo.ipynb)。

## 评测与基准测试 (Evals & Benchmarks)

### 网络安全 (Cybersecurity)

#### CyberSec Eval v1
CyberSec Eval v1 是业界首套针对大语言模型网络安全风险的体系化安全评测套件。该基准基于行业权威指引与标准（如 CWE 与 MITRE ATT&CK）构建，并与内部资深安全领域专家紧密协作完成。我们致力于提供实用工具以帮助应对[白宫负责任 AI 开发承诺声明](https://www.whitehouse.gov/briefing-room/statements-releases/2023/07/21/fact-sheet-biden-harris-administration-secures-voluntary-commitments-from-leading-artificial-intelligence-companies-to-manage-the-risks-posed-by-ai/)中指出的若干核心风险，包括：
* 量化大语言模型网络安全风险的评估指标；
* 评测 LLM 提出不安全代码建议频率的工具；
* 评估 LLM 生成恶意代码或协助网络攻击倾向的测试工具。

我们相信这些工具有助于降低 LLM 建议不安全代码的概率，并削弱其被网络攻击者滥用的可能。初步研究表明，大语言模型在建议不安全代码以及顺从恶意请求方面确实存在显著的网络安全风险。详见我们的 [CyberSec Eval 研究论文](https://ai.meta.com/research/publications/purple-llama-cyberseceval-a-benchmark-for-evaluating-the-cybersecurity-risks-of-large-language-models/)。

#### CyberSec Eval 2
CyberSec Eval 2 在前代基础上进行了全面扩充，重点衡量大语言模型滥用代码解释器的倾向、攻击性网络安全能力（Offensive Capabilities），以及对提示词注入攻击的抵御韧性。欢迎查阅[相关研究论文](https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/)。

您也可以访问位于 🤗 Hugging Face 的[官方评测榜单 (Leaderboard)](https://huggingface.co/spaces/facebook/CyberSecEval)。

#### CyberSec Eval 3
新发布的 CyberSec Eval 3 进一步引入了三套全新测试套件：视觉提示词注入测试（Visual Prompt Injection）、鱼叉式网络钓鱼能力测试（Spear Phishing），以及自主网络攻击行动测试（Autonomous Offensive Cyber Operations）。

## 快速上手 (Getting Started)

作为 [Llama 参考系统 (Llama Reference System)](https://github.com/meta-llama/llama-agentic-system) 的关键组成部分，我们正在深度整合安全防护层，以简化这些安全工具的采纳与落地部署。
更多上手安全防护工具的实战资源请参见 [Llama-recipes GitHub 仓库](https://github.com/meta-llama/llama-recipes)。

## 常见问题 (FAQ)

关于 Purple Llama 各组件以及 Llama 系列模型的常见问题整理与持续更新，请参阅[官方 FAQ](https://ai.meta.com/llama/faq/)。

## 加入 Purple Llama 社区

欢迎查阅 [CONTRIBUTING](CONTRIBUTING.md) 文档，了解如何参与项目贡献并共同构建更安全的开源 AI 生态。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月27日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
