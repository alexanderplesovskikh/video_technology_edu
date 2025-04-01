import random
import uuid
import logging
from datetime import datetime
import zulip
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import tqdm
from langchain_ollama.llms import OllamaLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import re
import torch
import gc
from langchain_core.prompts import ChatPromptTemplate

from db.events import store_event

import json
import os
from datetime import datetime

def escape_all_variables(template: str) -> str:
    """
    Escape all variables in the template string.

    Args:
    - template (str): The template string containing variables to escape.

    Returns:
    - str: The template string with all variables escaped.
    """
    # Use regex to replace {var} with {{var}} for any variable
    escaped_template = re.sub(r'\{([^}]+)\}', r'{{\1}}', template)
    return escaped_template

# Define the path to your JSON file
json_file_path = '/home/user/record_bot_data.json'

# Function to append data to the JSON file
def append_to_json_file(file_path, new_data):
    # Check if the file exists
    if os.path.exists(file_path):
        # Open the existing JSON file and load its content
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []  # If the file is empty or not valid JSON, start with an empty list
    else:
        # If the file does not exist, create an empty list
        data = []

    # Append the new data to the list
    data.append(new_data)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


gc.collect()
torch.cuda.empty_cache()

token_speed = 150

base_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

print('model start loading')
model_ollama = "gemma2:latest"
#best model but we need more memory
model_ollama = "electromagneticcyclone/t-lite-q:3_k_m"
model_ollama = "llama3.1:8b-instruct-q4_K_S"
model_ollama = "owl/t-lite:latest"

init_ollama = OllamaLLM(
    model=model_ollama,
    temperature = 0.3,
    streaming=True,
    num_ctx=4096,
    callbacks=[StreamingStdOutCallbackHandler()],
)

#max_tokens=10,
#num_predict=10,

print('model end loading')

user_level = ''
user_theme = ''
user_id = -1

def extract_number(file_name):
    # Use regex to find the number in the file name
    match = re.search(r'(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')  # Return a large number if no digit found

def check_folder_exists(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"The folder '{folder_path}' exists.")
        return True
    else:
        print(f"The folder '{folder_path}' does not exist.")
        return False

def get_chuncks(text_string, chunkssize, chunkoverlap, lengthfunc):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunkssize,
    chunk_overlap=chunkoverlap,
    length_function=lengthfunc,
    is_separator_regex=False,)

    texts = text_splitter.create_documents([text_string])

    res = [text.page_content for text in texts if len(text.page_content) > 300]
    return res

def read_first_lines(folder_path):
    first_lines = []
    all_lines = {}
    # Iterate through all files in the folder
    for file_name in sorted(os.listdir(folder_path), key=extract_number):
        # Check if the file has a .txt extension
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            # Open the file and read the first line
            with open(file_path, 'r', encoding='utf-8') as file:
                all_current_lines = file.readlines()
                if file_path in all_lines:
                    all_lines[file_path] += all_current_lines
                else:
                    all_lines[file_path] = all_current_lines


                first_line = all_current_lines[0].strip()  # Read and strip any trailing spaces/newlines
                print("file_path: ", file_path)
                print("first line: ", first_line)
                if first_line[0] == "#":
                    first_line = first_line[1:].strip()

                first_lines.append(first_line)

    return first_lines, all_lines

# Example usage
folder_path = base_dir + "course_materials"
lines, all_lines = read_first_lines(folder_path)

print(lines)
print(all_lines.keys())

lines_to_str = ''

all_theme_ids = []

for line_id in range(len(lines)):
    all_theme_ids.append(int(line_id+1))
    lines_to_str += str(line_id + 1) + ". " + lines[line_id] + "\n"


main_menu = f"""✋ Привет! Я бот-обучатор по Видеотехнологиям 🎥\n\n
Помогу тебе разобраться во всех деталях курса. **Для начала выбери тему, которая тебе интересна:**\n
{lines_to_str}
\n
**Отправь мне соответсвующий номер темы, например**: **```1```** или **```5```**\n
\n---\n
Чтобы вернуться в главное меню, введи: **`помощь`**
"""

# Example usage
folder_path_faiss = base_dir + "faiss"
folder_exists = check_folder_exists(folder_path_faiss)

vs_name = base_dir + 'faiss/faiss_1_1'
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

if not folder_exists:
    print('start to create faiss')
    res = {}
    for key in all_lines:
        temp_res = ''
        for itemm in all_lines[key]:
            temp_res += itemm
        if key in res:
            res[key] += temp_res
        else:
            res[key] = temp_res


    chunk_dict = {}
    for key in res:
        res_chunks = get_chuncks(res[key], 500, 50, len)
        if key in chunk_dict:
            chunk_dict[key] += res_chunks
        else:
            chunk_dict[key] = res_chunks

    chunk_key_pairs = []
    for key in chunk_dict:
        for chunk in chunk_dict[key]:
            chunk_key_pairs.append([key, chunk])



    embeddings = embedding_function.embed_documents('Это пример предложения.')
    index = faiss.IndexFlatL2(len(embeddings[0]))
    embed_dim = len(embeddings[0])
    vector_store = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    list_langchain_docs = []
    for element in tqdm.tqdm(chunk_key_pairs):
        key, chunk = element[0], element[1]
        list_langchain_docs.append(Document(page_content=chunk, metadata={"source": key},))

    uuids = [i+1 for i in range(len(list_langchain_docs))]

    print("uuids: ", uuids)

    vector_store.add_documents(documents=list_langchain_docs, ids=uuids)
    vector_store.save_local(vs_name)

new_vector_store = FAISS.load_local(
    vs_name, embedding_function, allow_dangerous_deserialization=True
)


def search_chunks(query,k=5):
    results = new_vector_store.similarity_search(
    query,
    k=k
    )

    gotten_chunks = []
    for res in results[:-1]:
        if " - " + res.page_content not in gotten_chunks:
            gotten_chunks.append(" - " + res.page_content)

    context_faiss = "\n".join(chunk for chunk in gotten_chunks)
    return context_faiss

def search_chunks_new_knowledge(query,k=5):
    results = new_vector_store.similarity_search(
    query,
    k=k
    )

    to_put = " - " + results[-1].page_content

    context_faiss = to_put

    return context_faiss 

def search_chunks_biblio(query,k=5):
    results = new_vector_store.similarity_search(
    query,
    k=k
    )

    gotten_chunks = []
    for res in results[:-1]:
        if " - " + res.page_content not in gotten_chunks:
            source_path = res.metadata['source']
            source_file = source_path.split("/")[-1].split(".txt")[0].strip()
            line_index = int(source_file) - 1
            source_line = lines[line_index]
            gotten_chunks.append(" - " + res.page_content + f"""\n*© Источник: {source_line}*\n---\n""")

    context_faiss = "\n".join(chunk for chunk in gotten_chunks)

    context_faiss = context_faiss.replace(":::", "")
    return context_faiss

def format_llm_prompt(query):
    response = init_ollama.invoke(query)
    return response
    #torch.cuda.empty_cache()
    #prompt = ChatPromptTemplate.from_template(repr(json.dumps(json.dumps(str(query)))))
    #chain = prompt | init_ollama
    #return chain.stream({'key': 'value'})


user_states = {}

user_history = {}

class QuizBot:
    def initialize(self, bot_handler):
        self.bot_handler = bot_handler
        self.setup_logging()
        self.client = zulip.Client(config_file=base_dir + "zuliprc")

    def setup_logging(self):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    def send_reply(self, message, response):
        res = store_event(recipient=message["sender_email"], content=response, operation_type='send')
        return res
    
    def update_reply(self, message, response, get_event_id):
        res = store_event(recipient=message['sender_email'], content=response, operation_type='update', updating_event_id=get_event_id)
        return res

    def usage(self) -> str:
        pass

    def handle_message(self, message, bot_handler):

        user_id = message["sender_id"]
        content = message["content"].strip().lower()

        if user_id not in user_history:
            user_history[user_id] = []

        if user_id not in user_states:
            user_states[user_id] = {"state": "main_menu"}

        state = user_states[user_id]["state"]

        if content in ["помощь", 'help', 'start', 'exit']:
            user_states[user_id] = {"state": "main_menu"}
            self.send_reply(message, main_menu)
            return

        if state == "main_menu":
            if re.match(r"^\d+$", content.strip()) and int(content.strip()) in all_theme_ids:
                user_states[user_id] = {"state": "select_level", "topic": int(content)}
                self.send_reply(message, "Какой у вас уровень знаний по теме? Выбери, введя соответствующую цифру: \n**`1. начальный`**\n**`2. средний`**\n**`3. продвинутый`**\n\nЧтобы вернуться в главное меню, введи: **`помощь`**")
            else:
                self.send_reply(message, main_menu)

        elif state == "select_level":
            if content in ["1", "2", "3"]:
                user_states[user_id] = {"state": "chat", "topic": user_states[user_id]["topic"], "level": content}
                #open file
                #prompt model to summorize it
                topic = user_states[user_id]["topic"]

                user_history[user_id] = []

                file_path_summary = base_dir + 'course_materials/' + str(topic) + ".txt"

                with open(file_path_summary, 'r', encoding="utf-8") as sum_file:
                    sum_lines = sum_file.readlines()

                sum_firstlines = sum_lines[0]

                if sum_firstlines[0] == "#":
                    sum_firstlines = sum_firstlines[1:].strip()

                sum_restlines = " ".join(sum_lines[1:])

                summary_prompt = f"""- Текст: {sum_restlines}.
- Задание: ты ИИ-ассистент, напиши короткое резюме данного тебе содержания темы курса, кратко по пунктам обозначь основные разделы, через знак "-".
- Не пиши много текста, просто приведи подтемы данного тебе материала по курсу, через "-", каждый пункт начинай с новой строки.
- В конце резюме с нового абзаца, спроси про что-то одно из резюме - задай открытый вопрос по типу: А знаешь ли ты, что такое ..."""

                self.send_reply(message, "👨‍🏫 Давай посмотрим, что нам предстоит узнать в рамках этой темы.")

                llm_summary = format_llm_prompt(summary_prompt)

                user_states[user_id]['init_message'] = ""
                user_states[user_id]['last_updated_messsage'] = ""


                user_states[user_id]['init_message'] = f"Проведу тебе краткий экскурс по теме: **{sum_firstlines}**\n---\n"

                event_id = self.send_reply(message, user_states[user_id]['init_message'])

                if event_id:
                    count = 0
                    for token in llm_summary:
                        user_states[user_id]['init_message'] += token
                        count+= 1 
                        if count % token_speed == 0:
                            self.update_reply(message, user_states[user_id]['init_message'], event_id)
                            user_states[user_id]['init_message'] = user_states[user_id]['init_message']
                            
                    
                    if user_states[user_id]['init_message'] != user_states[user_id]['last_updated_messsage']:
                        self.update_reply(message, user_states[user_id]['init_message'], event_id)

                    ending_message = "\n---\n**❓ Давай начнём изучение темы! Можешь задать вопрос по любому аспекту темы, я отвечу с опорй на материалы курса Сетевых видеотехнологий или можем начать беседу по предложенному мной вопросу — в любом случае выбирать тебе! Просто `напиши свой вопрос` или отправь мне: `Расскажи, что такое <...>` — и мы начнем беседу 😉**\n---\nДля возврата в меню введите **`помощь`**."
                    self.update_reply(message, user_states[user_id]['init_message'] + ending_message, event_id)

                    if len(user_history[user_id]) > 10:
                        del user_history[user_id][0]

                    user_history[user_id].append({"role": "AI-assistant", "message": user_states[user_id]['init_message']})
            
            else:
                self.send_reply(message, "Пожалуйста, выберите уровень, введя соответствующую цифру: \n**`1. начальный`** \n**`2. средний`** \n**`3. продвинутый`**\n\nЧтобы вернуться в главное меню, введи: **`помощь`**")

        elif state == "chat":
            topic = user_states[user_id]["topic"]
            level = user_states[user_id]["level"]
            topic_name = lines[int(str(topic).strip())-1]

            model_level_desc = ''
            if str(level).strip() == '1':
                model_level_desc = 'объяснения попроще с простой лексикой и покроче'
            if str(level).strip() == '2':
                model_level_desc = 'средний уровень сложности лексики и средний уровень сложности объяснений'
            if str(level).strip() == '3':
                model_level_desc = 'сложный уровень лексики научный академический формальный язык и продвинутые объяснения'

            #Main answer

            chunks_for_llm = search_chunks(content)

            user_query = f"""# Помни, что есть история разговора: {str(user_history[user_id])}

# Контекст для ответа:
- Контекст и информация по курсу: {chunks_for_llm}.

# Вводные данные:
- Ты преподователь-профессор в Высшей школе экономики и к тебе обращаются студенты по теме курса "{topic_name}".
- Ты должен ответить на вопрос студенту или просто поддержать беседу по теме курса, если вопроса нет.
- Студент тебе написал: {content}.
- Для ответа используй {model_level_desc}.
- ПОМНИ, что нужно обязательно использовать контекст и информацию, предоставленную тебе по курсу.
- Длина твоего ответа и текста не должны быть больше 2-3 абзацев.
# Структура ответа:
- Ответь на вопрос студента (в рамках курса) или поддержки разговор со студентом (в рамках курса) самым лучшим образом, последовательно и логично.
- Твой ответ НЕ ДОЛЖЕН ПРЕВЫШАТЬ по объему 2-3 абзацев.
"""

            chunks_get = search_chunks_biblio(content)

            #user_states[user_id]['init_message'] = f"**Debug info:** вы выбрали тему {topic}. {topic_name}. C уровнем знаний: {level}. Ваш вопрос: {content}\n\n💭...\n\n"
            user_states[user_id]['init_message'] = f"💭...\n\n"
            user_states[user_id]['last_updated_messsage'] = ""

            event_id = self.send_reply(message, user_states[user_id]['init_message'])
            llm_main_response = format_llm_prompt(user_query)

            user_states[user_id]['to_be_checked_by_agent'] = ''

            if event_id:
                count = 0
                for token in llm_main_response:
                    user_states[user_id]['init_message'] += token
                    count+= 1 
                    user_states[user_id]['to_be_checked_by_agent'] += token
                    if count % token_speed == 0:
                        self.update_reply(message, user_states[user_id]['init_message'], event_id)
                        user_states[user_id]['init_message'] = user_states[user_id]['init_message']
                            
                    
                if user_states[user_id]['init_message'] != user_states[user_id]['last_updated_messsage']:
                    self.update_reply(message, user_states[user_id]['init_message'], event_id)

                
                response_ending = f"""\n```spoiler  📚 Источники информации:
{chunks_get}
```
\n
"""
                self.update_reply(message, user_states[user_id]['init_message'] + response_ending, event_id)

                if len(user_history[user_id]) > 10:
                    del user_history[user_id][0]

                user_history[user_id].append({"role": "AI-assistant", "message": user_states[user_id]['init_message']})
                

            else:
                self.send_reply(message, "Fatal error")

            #Agents start

            user_states[user_id]['init_message'] = "✅ Подключаем агентов по проверке ответов...\n"
            user_states[user_id]['last_updated_messsage'] = ""

            event_id = self.send_reply(message, user_states[user_id]['init_message'])

            agent_prompt = f"""### Помни, что есть история разговора: {str(user_history[user_id])}
            
### Текст: 
{user_states[user_id]['to_be_checked_by_agent']}
### Информация, на основе которой должен быть текст:
{chunks_for_llm}

### Задание:
- Ты ассистент по курсу компьютерная графика. Следуй следующей инструкции:
- Необходимо, чтобы ты проверил текст на его соответсвие информации. Проверь, что текст был написан с опорой исключительно на данную информацию. Если текст не соответсвует заданной информации, то ты должен исправить текст так, чтобы в нем была использована ТОЛЬКО предоставленная информация и НИКАКОЙ ДРУГОЙ ВНЕШНЕЙ ИНФОРМАЦИИ.
- Структура ответа должна быть следующей: в ответе напиши ТОЛЬКО исправленный текст с опорой только на данную тебе информацию.
- Текст является ответом на следующий вопрос: {content}.
- Исправленный текст НЕ должен превышать по объему 2-3 абзацев.
- Отправь мне ТОЛЬКО исправленный текст, не приводи мне никаких рассуждений.
"""

            llm_agent = format_llm_prompt(agent_prompt)

            user_states[user_id]['agents_get_response'] = ''

            if event_id:
                count = 0
                for token in llm_agent:
                    user_states[user_id]['init_message'] += token
                    user_states[user_id]['agents_get_response'] += token
                    count+= 1 
                    if count % token_speed == 0:
                        self.update_reply(message, user_states[user_id]['init_message'], event_id)
                        user_states[user_id]['init_message'] = user_states[user_id]['init_message']
                            
                    
                if user_states[user_id]['init_message'] != user_states[user_id]['last_updated_messsage']:
                    self.update_reply(message, user_states[user_id]['init_message'], event_id)

                '''agent_mes_ending = f"""\n```spoiler  ⛔ Debug info:
{agent_prompt}
```
"""'''
                agent_mes_ending = ""

                self.update_reply(message, user_states[user_id]['init_message'] + agent_mes_ending, event_id)

                if len(user_history[user_id]) > 10:
                    del user_history[user_id][0]

                user_history[user_id].append({"role": "AI-assistant", "message": user_states[user_id]['init_message']})
            
            else:
                self.send_reply(message, "Fatal error")

            #Question back
            user_states[user_id]['init_message'] = "📝 Формирую план обсуждения...\n"
            user_states[user_id]['last_updated_messsage'] = ""

            event_id = self.send_reply(message, user_states[user_id]['init_message'])

            new_know_chunks = search_chunks_new_knowledge(content)
            new_query = f"""### Вот информация: {new_know_chunks}. ### Задание: сформулируй один открытый вопрос по данной тебе информации, в ответе укажи только сам вопрос. Не обязательно пытаться использовать все факты из информации, можно ограничиться лишь частью информации и по ней сформулировать один открытый вопрос. Вопрос начни со слов: "А знаешь ли ты", затем твой вопрос и в конце уточни: "если не знаешь, я обязательно тебе расскажу!" """
            llm_new_question = format_llm_prompt(new_query)

            if event_id:
                count = 0
                for token in llm_new_question:
                    user_states[user_id]['init_message'] += token
                    count+= 1 
                    if count % token_speed == 0:
                        self.update_reply(message, user_states[user_id]['init_message'], event_id)
                        user_states[user_id]['init_message'] = user_states[user_id]['init_message']
                            
                    
                if user_states[user_id]['init_message'] != user_states[user_id]['last_updated_messsage']:
                    self.update_reply(message, user_states[user_id]['init_message'], event_id)

                '''new_mes_ending = f"""\n```spoiler  ⛔ Debug info:
{new_query}
```
"""'''
                new_mes_ending = ""

                self.update_reply(message, user_states[user_id]['init_message'] + new_mes_ending, event_id)

                #base llm without rag and agent

                base_user_query = f"""#Помни, что есть история разговора: {str(user_history[user_id])}
# Вводные данные:
- Ты преподователь-профессор в Высшей школе экономики и к тебе обращаются студенты по теме курса "{topic_name}".
- Ты должен ответить на вопрос студенту или просто поддержать беседу по теме курса, если вопроса нет.
- Студент тебе написал: {content}.
- Для ответа используй {model_level_desc}.
- ПОМНИ, что нужно обязательно использовать контекст и информацию, предоставленную тебе по курсу.
- Длина твоего ответа и текста не должны быть больше 2-3 абзацев.
# Структура ответа:
- Ответь на вопрос студента (в рамках курса) или поддержки разговор со студентом (в рамках курса) самым лучшим образом, последовательно и логично.
- Твой ответ НЕ ДОЛЖЕН ПРЕВЫШАТЬ по объему 2-3 абзацев.
"""
                
                user_states[user_id]['base_response'] = ''

                base_llm = format_llm_prompt(base_user_query)

                for token in base_llm:
                    user_states[user_id]['base_response'] += token

                
                # Define the new data you want to add
                new_entry = {
                    'time': datetime.now().isoformat(),
                    'user_id': str(message["sender_email"]),  # Replace with actual user ID
                    'question': str(content),
                    'rag_chunks': str(chunks_for_llm),
                    'theme': str(topic),  # Replace with actual theme
                    'theme_name': str(topic_name),  # Replace with actual theme
                    'level': str(level),  # Replace with actual level
                    'llm_base': str(user_states[user_id]['base_response']),  # Replace with actual LLM base
                    'llm_rag': str(user_states[user_id]['to_be_checked_by_agent']),  # Replace with actual LLM RAG
                    'llm_agent': str(user_states[user_id]['agents_get_response'])  # Replace with actual LLM agent
                }

                # Call the function to append data
                append_to_json_file(json_file_path, new_entry)





handler_class = QuizBot