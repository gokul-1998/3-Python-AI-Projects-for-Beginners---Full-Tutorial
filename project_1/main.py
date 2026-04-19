import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set. Please add it to your .env file.")
        return

    # Use ChatOpenAI but point it to GitHub Models by overriding base_url and api_key
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=token,
        base_url="https://models.github.ai/inference"
    )
    
    tools = []
    agent_executor = create_react_agent(model, tools)
    
    print("Welcome! I'm your AI assistant. Type quit to exit")
    print("You can ask me to perform calculations or chat with me")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
            
        print("\nAssistant: ", end="")
        try:
            # LangGraph streaming
            for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
                if "agent" in chunk and "messages" in chunk['agent']:
                    for msg in chunk['agent']['messages']:
                        if msg.type == "ai":
                            print(msg.content, end="")
            print()
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()