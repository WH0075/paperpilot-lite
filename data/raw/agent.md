# Agent

An AI agent is a system that can perceive information, reason about a task, and take actions using tools. In modern LLM-based applications, an agent is usually built around a large language model, but the language model itself is not the whole agent. The agent also needs tools, memory, planning logic, and a way to observe the results of its actions.

A typical LLM agent follows a loop such as Thought, Action, Observation, and Answer. The model first analyzes the user request, decides whether it needs a tool, calls the selected tool, observes the returned result, and then continues reasoning. This loop allows the agent to solve tasks that cannot be answered from model parameters alone.

Tools are important because they extend the capability of the language model. Common tools include search engines, calculators, code interpreters, databases, APIs, and RAG systems. A calculator is useful for exact arithmetic, a code interpreter is useful for executing programs, and a RAG system is useful when the agent needs external knowledge from local documents.

RAG can serve as a knowledge tool for agents. When an agent receives a question about private documents, project files, papers, or company policies, it can call a RAG tool to retrieve relevant chunks. The retrieved chunks become evidence for the final answer. This reduces hallucination and makes the agent more grounded.

A good agent system should also record its intermediate steps. The record of tool calls, observations, and final answers is often called an agent trace. Agent traces are useful for debugging because they show whether the model selected the right tool, whether the retrieved evidence was relevant, and where the reasoning process failed.

Agents can fail in several ways. They may choose the wrong tool, call the right tool with a bad query, ignore useful observations, or produce an answer that is not supported by evidence. For this reason, agent evaluation should measure not only the final answer, but also tool selection accuracy, evidence quality, and task completion rate.
