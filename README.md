# Web-Voyager-Using-Langchain
This is a vision-enabled web-browsing agent capable of controlling the mouse and keyboard.<br>
It extends the Langchain Expression Language with the ability to coordindate multiple chain(or actors) across multiple steps of computation in a cyclic manner( adding cycles to your LLM application).<br>
Cycles are important for agent-like behaviors, you call the llm in a loop, asking it what action to take next.

It works by viewing annotated browser screenshots for each turn, then choosing the next step to take.The agent architecure is basic reasoning and action (ReAct) loop. The unique aspects of this  agent are:
- Its usage of ```set-of-marks``` like image to serve as UI affordances for the agent
- Its application in the browser by using tools to control both the mouse and keyboard

  The overall desing look like the following

  ![web-voyager excalidraw](https://github.com/Kevin7744/Web-Voyager-Using-Langchain/assets/105924200/37bb481b-1594-419a-a1dc-f87691e36af1)


Install packages
```
pip install langchain langchain-core langchain_openai langgraph playwright
```

