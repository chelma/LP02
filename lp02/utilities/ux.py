from json import dumps
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

def simplify_history(raw_messages: List[BaseMessage]) -> List[Dict[str, any]]:
    simplified_history = []
    for message in raw_messages:
        if isinstance(message, HumanMessage):
            simplified_history.append({
                "type": "User Entry",
                "content": message.content                
            })
        elif isinstance(message, SystemMessage):
            simplified_history.append({
                "type": "System Entry",
                "content": message.content
            })
        elif isinstance(message, ToolMessage):
            simplified_history.append({
                "type": "Tool Entry",
                "tool_name": message.name,
                "content": message.content                
            })
        elif isinstance(message, AIMessage):
            entry = {
                "type": "AI Entry"
            }

            if isinstance(message.content, str):
                entry["content"] = message.content
            elif isinstance(message.content, list):
                for item in message.content:
                    if item["type"] == "text":
                        entry["content"] = item["text"]
                    elif item["type"] == "tool_use":
                        tool_uses = entry.get("tool_use", [])
                        tool_uses.append({
                            "tool_name": item["name"],
                            "input": item["input"]
                        })
                        entry["tool_use"] = tool_uses
            else:
                entry["content"] = str(message.content)
            simplified_history.append(entry)
    return simplified_history

def stringify_simplified_history(raw_messages: List[BaseMessage]) -> str:
    simplified_history = simplify_history(raw_messages)
    return dumps(simplified_history, indent=4)
