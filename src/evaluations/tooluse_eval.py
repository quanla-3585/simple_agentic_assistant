from agentic_sun_assistant.graph import create_and_compile_graph
from langchain_core.messages import AIMessage
import asyncio
import dotenv

dotenv.load_dotenv()

eval_graph = create_and_compile_graph()

message_trace = asyncio.run(
    eval_graph.ainvoke(
        {
            "type":"human", "content":"Where is the FooFirm employee handbook stored?"
        }
    )
)

for message in message_trace:
    if isinstance(message, AIMessage):
        try:
            print(message.tool_calls)
        except:
            print("None")

# print([msg_.__class__ for msg_ in message_trace["messages"]])