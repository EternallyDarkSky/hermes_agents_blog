---
title: "AI Agent 的核心特点:自主性、工具调用、记忆与上下文、规划反思、多智能体协作"
date: 2026-09-18
draft: false
tags: ["AI Agent", "LLM", "Autonomy", "Tool Use", "Memory", "Planning", "Multi-Agent"]
author: "丁志强"
---

一个 LLM 本身只会「一次前向生成」:你给它一段 prompt,它吐出一段文本,然后结束。那一个能自己订机票、改代码、来回排查问题的 AI Agent,和它到底差在哪?答案是:Agent 在模型之上,又加了一个可以循环的运行时,由模型自己决定下一步做什么。[3] 这个循环有五个侧面——自主性、工具调用、记忆与上下文、规划反思、多智能体协作。它们不是五个并列的功能,而是同一个循环的五个面:自主性是循环本身,工具调用是循环的手脚,记忆与上下文是循环的桌面,规划与反思是循环的调度,多智能体协作是循环的横向扩展。[3][13]

## 一、自主性:从「按流程跑」到「自己拿主意」

写 Agent 前,先想清楚自己到底要不要「自主」。Anthropic 把二者都归为 agentic systems,但划了一条清晰的线:workflow 是「LLM 与工具由预定义代码路径编排」,agent 则是「LLM 动态决定自己的流程与工具使用」[3]。这是架构选择的第一个岔路口,选错就白白付出延迟和成本。客服场景里,「按固定顺序查订单、查知识库、再回复」做成 workflow 即可;只有工单内容不确定、需要临场决定查哪个系统、甚至要发起退款时,才值得做成 agent。[3]

自主性的技术形态,就是一个「感知—决策—行动」的循环:agent 在循环里依据环境反馈调用工具,直到任务完成或命中停止条件[3]。ReAct 让模型交错生成推理轨迹与动作,在 HotpotQA/Fever 上通过与 Wikipedia API 交互缓解了 chain-of-thought 的幻觉,并在 ALFWorld、WebShop 上分别取得 34% 与 10% 的绝对成功率提升,且只用了一两个 in-context 示例[1]。但循环也有代价:官方数据显示 agent 的 token 消耗约为普通对话的 4 倍,多智能体系统约为 15 倍,所以「上不上 agent」本质是一道经济题,而不是能力题[13]。工程上还必须给自主性装刹车——用检查点、停止条件,以及并行运行的 Guardrails 在失败时快速停下并留出 human in the loop 的入口。[3][17]

## 二、工具调用:给模型装上手脚和接口

工具调用(Tool Use / Function Calling)的本质,是「模型只出结构化意图,执行留在应用侧」。官方定义是:Claude 根据请求与工具描述自行决定何时调用工具,并返回结构化调用,由应用(client tools)或平台(server tools)落地执行[15]。这条边界把权限、凭据与副作用都留在你可控的代码里,而不是塞进模型的嘴里。一个最小的闭环长这样:

```python
import json

tools = [{
    "type": "function",
    "name": "web_search",
    "description": "搜索外部网页,输入一个查询字符串,返回标题与摘要列表。仅当需要外部事实时调用。",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"}},
                   "required": ["query"]},
}]

messages = [{"role": "user", "content": "查一下最新的 ReAct 论文"}]

while True:
    resp = model.chat(messages=messages, tools=tools)  # 模型只出意图
    if not resp.tool_calls:                            # 不再请求工具,结束
        break
    for call in resp.tool_calls:
        result = execute_local(call.name, json.loads(call.arguments))
        messages.append({"role": "tool", "content": result})  # 结果回填
    messages.append(resp.message)                       # 继续循环
```

注意那个 `description` 字段:它本身就是 prompt 的一部分。Anthropic 把 agent–computer interface(ACI)提到与人机界面同等的高度,一条写坏的描述足以让 agent 完全跑偏;他们把有缺陷的 MCP 工具交给一个「工具测试 agent」反复试用并重写描述后,后续任务完成时间下降了 40%[13]。工具能力还可以被模型自己习得:Toolformer 让语言模型以自监督方式学会「调哪个 API、何时调、传什么参数」,只用每个 API 的少量示例,就在多项下游任务上显著提升 zero-shot 表现,常常能与更大的模型竞争[2]。当工具数量多起来,MCP(Model Context Protocol)把它变成「AI 应用的 USB-C」,一次构建、处处集成。[14]

## 三、记忆与上下文:窗口不是硬盘,而是天天整理的桌面

官方把 context window 定义为模型生成时可引用的全部文本,是模型的 working memory——system prompt、messages、工具结果、图片文档、甚至模型自己的 thinking 都算数[10]。把它当成无限硬盘,是 agent 长任务崩溃的头号原因。更反直觉的是「塞得多」不等于「记得住」:needle-in-a-haystack 类评测揭示的 context rot 现象说明,上下文 token 越多,模型准确召回其中信息的能力越差[9]。所以上下文管理从「扩容问题」变成了「取舍问题」,Anthropic 由此从 prompt engineering 转向 context engineering——把「为下一次推理挑选哪些 token」当作每次都要重做的整理动作。[9]

当窗口快满时,工程解法是分层记忆:把本轮要点写入外部长期记忆,再压缩工作集,只留摘要与检索引用,下一轮按需取回。这正是 MemGPT 借鉴操作系统虚拟内存的思路,在快慢内存之间搬运数据,用有限的窗口提供「更大的记忆」[8]:

```python
def maybe_compress(memory, window_limit):
    if memory.usage() > window_limit:
        summary = summarize(memory.working_set)      # 阶段总结
        memory.long_term.append(summary)             # 写入外部长期记忆
        memory.working_set = [summary]               # 只留摘要与引用
    return memory

def answer(q):
    memory.working_set += retrieve(q, memory.long_term)  # 按需取回
    return model.chat(memory.working_set)
```

这套「记录经验 → 反思为高层结论 → 按需检索驱动规划」的闭环,在 Generative Agents 里被验证过:一个 25 个 agent 的沙盒小镇,仅凭用户设定「某人想办情人节派对」这一条信息,agent 们就在随后两天自发邀请、结识、互相邀约并协调赴约[7]。生产环境里,Anthropic 更进一步:让 agent 总结已完成阶段、把关键信息存入外部记忆,接近上限时用带干净上下文的新 subagent 接班,只把轻量引用回传给协调者,以减少多级传递的信息损耗。[13]

## 四、规划与反思:先想、再做、再改

规划的本质,是把「一条直线」变成「可搜索的树」。Tree of Thoughts 把 chain-of-thought 泛化为对多个候选思路的探索:模型可以同时考虑几条推理路径、自我评估后决定下一步,必要时前瞻或回溯[5]。它针对的正是「初始决策很关键、需要探索」的任务:在 Game of 24 上,GPT-4 配 chain-of-thought 只解决 4%,ToT 达到 74%[5]。反思则用「语言反馈」替代「权重更新」:Reflexion 不更新参数,而是让 agent 对失败信号写一段反思文本,存进 episodic memory,并在下一次尝试时带上它,从而在没有微调的情况下从试错中学习[4]:

```python
def solve(task, max_tries=3):
    memory = []
    for _ in range(max_tries):
        trail = model.act(task, memory)              # 带上历史反思再试
        score, feedback = environment.score(trail)
        if score >= task.threshold:
            return trail
        memory.append(model.reflect(trail, feedback))  # 反思存进 episodic buffer
    return None
```

Reflexion 在 HumanEval 上拿到 91% 的 pass@1,超过当时 GPT-4 的 80%[4]。这条线再往下走,就是 Voyager 的「技能库」:把环境反馈、执行错误与自我验证都整合进程序改进,把学到的技能以可执行代码形式存起来、可积累可迁移[6]。落到工程上,Anthropic 把规划归纳成三种可直接选型的 workflow——prompt chaining、orchestrator-workers、evaluator-optimizer,让「规划」从抽象概念变成结构。[3]

## 五、多智能体协作:如何分工,以及什么时候不该分工

多智能体的第一性收益不是「人多力量大」,而是「并行压缩」:每个 subagent 拥有独立的上下文窗口,并行探索问题的不同侧面,再把最重要的 token 压缩回主 agent,同时带来分离的关注点、降低路径依赖[13]。Anthropic 内部评测里,Claude Opus 4 牵头、Sonnet 4 当 subagent 的系统,比单 agent 的 Opus 4 高出 90.2%[13]。但「加 agent」要算账:token 用量本身解释了 BrowseComp 评测约 80% 的性能方差,而多智能体系统的 token 消耗约为普通对话的 15 倍,只有任务价值足够高时才经济可行[13]。

更重要的是,多智能体不是万能药。需要所有 agent 共享同一上下文、或 agent 间强依赖的任务,目前就不适合——例如多数编码任务真正可并行的部分少于研究任务[13]。要压住级联幻觉,得靠组织结构而非堆模型:MetaGPT 把 SOP(标准作业程序)编码进 prompt 序列、给不同 agent 分配角色并校验中间结果[12];落地层则有两条路——OpenAI Agents SDK 的 handoffs(agent 把任务委派给别的 agent)与 LangGraph 的图编排(节点做事、边决定下一步)[17][20]。生产化的代价同样现实:agent 有状态、错误会复利,需要全链路 tracing 与可恢复的中断处,原型到生产的差距往往比预期更大。[13]

## 小结

回头看这五个特点,会发现它们其实是同一件事的不同侧面:一个可以循环的运行时,自己决定下一步、用手脚去执行、在桌面上整理记忆、靠调度来规划反思、必要时横向扩展出去。这也给出了实践上的起点——不要一上来就堆多智能体,先从最小可用的 workflow 开始,只在任务真的需要动态决策、真的超出单个上下文窗口时,才逐层增加复杂度。[3][13]

## 参考资料

- [1] ReAct: Synergizing Reasoning and Acting in Language Models — https://arxiv.org/abs/2210.03629
- [2] Toolformer: Language Models Can Teach Themselves to Use Tools — https://arxiv.org/abs/2302.04761
- [3] Building Effective Agents — https://www.anthropic.com/engineering/building-effective-agents
- [4] Reflexion: Language Agents with Verbal Reinforcement Learning — https://arxiv.org/abs/2303.11366
- [5] Tree of Thoughts — https://arxiv.org/abs/2305.10601
- [6] Voyager: An Open-Ended Embodied Agent with Large Language Models — https://arxiv.org/abs/2305.16291
- [7] Generative Agents: Interactive Simulacra of Human Behavior — https://arxiv.org/abs/2304.03442
- [8] MemGPT: Towards LLMs as Operating Systems — https://arxiv.org/abs/2310.08560
- [9] Effective Context Engineering for AI Agents — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- [10] Context Windows — https://docs.claude.com/en/docs/build-with-claude/context-windows
- [12] MetaGPT: Meta Programming for Multi-Agent Collaborative Framework — https://arxiv.org/abs/2308.00352
- [13] How We Built Our Multi-Agent Research System — https://www.anthropic.com/engineering/multi-agent-research-system
- [14] Model Context Protocol — https://modelcontextprotocol.io
- [15] Tool Use (Function Calling) Overview — https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview
- [17] OpenAI Agents SDK — https://openai.github.io/openai-agents-python
- [20] Graph API Overview (State/Nodes/Edges) — https://docs.langchain.com/oss/python/langgraph/graph-api
