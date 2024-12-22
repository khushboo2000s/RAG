# # import json
# # import os

# # # Memory file path
# # MEMORY_FILE = "query_memory.json"

# # # Function to load the memory file
# # def load_memory():
# #     if os.path.exists(MEMORY_FILE):
# #         with open(MEMORY_FILE, "r") as f:
# #             try:
# #                 return json.load(f)
# #             except json.JSONDecodeError:
# #                 return {}
# #     return {}

# # # Function to save data to the memory file
# # def save_to_memory(data):
# #     with open(MEMORY_FILE, "w") as f:
# #         json.dump(data, f, indent=4)

# # # Function to update memory with a new query result
# # def update_memory(query, sources, answer):
# #     memory_data = load_memory()

# #     memory_data[query] = {
# #         "question": query,
# #         "sources": sources,
# #         "answer": answer
# #     }

# #     save_to_memory(memory_data)

# # # Function to check if a query result is already in memory
# # def get_from_memory(query):
# #     memory_data = load_memory()
# #     return memory_data.get(query, None)







# ############## Chat-with-memory ##################################
# ##################################################################
# # import json
# # import os

# # # Memory file path
# # MEMORY_FILE = "query_memory.json"

# # # Function to load the memory file
# # def load_memory():
# #     if os.path.exists(MEMORY_FILE):
# #         with open(MEMORY_FILE, "r") as f:
# #             try:
# #                 memory_data = json.load(f)
# #                 if isinstance(memory_data, dict):
# #                     # If the loaded data is a dictionary, convert it to a list
# #                     return [memory_data]
# #                 return memory_data
# #             except json.JSONDecodeError:
# #                 return []  # Return empty list on decode error
# #     return []  # Return empty list if the file doesn't exist

# # # Function to save data to the memory file
# # def save_to_memory(data):
# #     with open(MEMORY_FILE, "w") as f:
# #         json.dump(data, f, indent=4)

# # # Function to update memory with a new query result (no user_id)
# # # Function to update memory with a new query result (no user_id)
# # def update_memory(query, sources, answer):
# #     memory_data = load_memory()
    
# #     if not isinstance(memory_data, list):  # Ensure it's a list
# #         memory_data = []

# #     memory_data.append({
# #         "question": query,
# #         "sources": sources,
# #         "answer": answer
# #     })

# #     save_to_memory(memory_data)


# # # Function to get chat history (no user_id)
# # def get_chat_history():
# #     memory_data = load_memory()
# #     return [entry for entry in memory_data if isinstance(entry, dict)]  # Return only dictionaries


# # # Function to retrieve the most recent interaction for context
# # def get_last_answer():
# #     history = get_chat_history()
# #     return history[-1] if history else None

# # # Function to retrieve a specific query result from the memory file
# # def get_from_memory(query):
# #     memory_data = load_memory()  # Load the memory data
# #     for entry in memory_data:
# #         if isinstance(entry, dict) and entry.get('question') == query:  # Ensure entry is a dict
# #             return entry  # Return the entire entry if it matches the query
# #     return None  # Return None if no match is found



# ########################################################################################

# ######################### Testing-1 ########################################
# ###############################################################################
# import json
# import os

# # Memory file path
# MEMORY_FILE = "query_memory.json"

# # Function to load the memory file
# def load_memory():
#     """Load memory from file, ensure it's a list."""
#     if os.path.exists(MEMORY_FILE):
#         with open(MEMORY_FILE, "r") as f:
#             try:
#                 memory_data = json.load(f)
#                 if isinstance(memory_data, list):
#                     return memory_data  # Ensure it's a list
#                 else:
#                     return []  # If not a list, return an empty list
#             except json.JSONDecodeError:
#                 return []  # Return empty list on JSON decode error
#     return []  # Return empty list if the file doesn't exist

# # Function to save data to the memory file
# def save_to_memory(data):
#     """Save data to memory file."""
#     with open(MEMORY_FILE, "w") as f:
#         json.dump(data, f, indent=4)

# # Function to update memory with a new query result (no user_id)
# def update_memory(query, sources, answer):
#     """Add a new query result to memory."""
#     memory_data = load_memory()

#     # Ensure memory_data is always a list
#     if not isinstance(memory_data, list):
#         memory_data = []

#     memory_data.append({
#         "question": query,
#         "sources": sources,
#         "answer": answer
#     })

#     save_to_memory(memory_data)

# # Function to update chat history with context
# def update_chat_history(question, answer, sources):
#     """Update chat history with the latest question, answer, and sources."""
#     memory_data = load_memory()
    
#     # Ensure the memory data is a list and check for existing 'context'
#     chat_history = {"context": ""}
#     if memory_data:
#         chat_history = memory_data[0] if isinstance(memory_data[0], dict) else {"context": ""}

#     # Append new conversation to the context
#     new_entry = f"Question: {question}\nAnswer: {answer}\nSources: {sources}\n"
#     chat_history['context'] += "\n" + new_entry

#     # Save updated chat history to memory (overwrite the first entry)
#     save_to_memory([chat_history])

# # Function to safely retrieve chat history context
# def get_chat_history():
#     """Retrieve the chat history context if it exists."""
#     memory_data = load_memory()
#     # Check if the memory data has any entries
#     if not memory_data:
#         return ""  # Return empty string if no data

#     # Ensure that the first entry has a 'context' key
#     first_entry = memory_data[0]
#     if isinstance(first_entry, dict) and 'context' in first_entry:
#         return first_entry['context']  # Return the context if present

#     return ""  # Return empty string if no context is found

# # Function to retrieve the most recent interaction for context
# def get_last_answer():
#     """Retrieve the most recent interaction from the chat history."""
#     history = get_chat_history()
#     return history.strip().split("\n")[-1] if history else None

# # Function to retrieve a specific query result from the memory file
# def get_from_memory(query):
#     """Get a specific query result by searching the memory."""
#     memory_data = load_memory()
#     for entry in memory_data:
#         if isinstance(entry, dict) and entry.get('question') == query:
#             return entry  # Return the matching entry if found
#     return {"context": ""}  # Return a default structure if no match is found






################### New - Approach ####################################
#######################################################################
import json
import os

MEMORY_FILE = 'memory.json'
CHAT_HISTORY_FILE = 'chat_history.json'

# Initialize memory storage if not already present
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, 'w') as f:
        json.dump({}, f)

if not os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump([], f)

def get_from_memory(question):
    """Retrieve an answer from memory based on the question."""
    with open(MEMORY_FILE, 'r') as f:
        memory = json.load(f)

    return memory.get(question)

def update_memory(question, sources, answer):
    """Update memory with a new question, sources, and answer."""
    with open(MEMORY_FILE, 'r') as f:
        memory = json.load(f)

    # Store the answer along with the sources
    memory[question] = {
        'answer': answer,
        'sources': sources
    }

    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f)

def get_chat_history():
    """Retrieve the chat history."""
    with open(CHAT_HISTORY_FILE, 'r') as f:
        chat_history = json.load(f)
    return chat_history

def add_to_chat_history(user_message, assistant_response):
    """Add a new message to chat history."""
    # Retrieve existing chat history
    with open(CHAT_HISTORY_FILE, 'r') as f:
        chat_history = json.load(f)

    # Append the new messages
    chat_history.append({
        'user_message': user_message,
        'assistant_response': assistant_response
    })

    # Save the updated chat history back to the file
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(chat_history, f)

def clear_memory():
    """Clear the memory file."""
    with open(MEMORY_FILE, 'w') as f:
        json.dump({}, f)

def clear_chat_history():
    """Clear the chat history."""
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump([], f)

##############################################################################