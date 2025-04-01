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
gc.collect()
torch.cuda.empty_cache()

base_dir = "/home/user/zulip-videoedu/python-zulip-api/zulip_bots/zulip_bots/bots/videoedu/"

print('model start loading')
model_ollama = "gemma2:latest"
#best model but we need more memory
model_ollama = "electromagneticcyclone/t-lite-q:3_k_m"
model_ollama = "llama3.1:8b-instruct-q4_K_S"
model_ollama = "owl/t-lite:latest"

init_ollama = OllamaLLM(
    model=model_ollama,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
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
    return [text.page_content for text in texts]

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
Отправь мне соответсвующий номер темы, например: **```1```** или **```5```**\n\n
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
    return context_faiss

def format_llm_prompt(query):
    response = init_ollama.invoke(query)
    #torch.cuda.empty_cache()
    return response


user_states = {}

class QuizBot:
    def initialize(self, bot_handler):
        self.bot_handler = bot_handler
        self.setup_logging()
        self.client = zulip.Client(config_file=base_dir + "zuliprc")

    def setup_logging(self):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    def send_reply(self, message, response):
      
        self.client.send_message({
            "type": "private",
            "to": message["sender_email"],
            "content": response,
        })

    def usage(self) -> str:
        pass

    def handle_message(self, message, bot_handler):
    
        user_id = message["sender_id"]
        content = message["content"].strip().lower()

        if user_id not in user_states:
            user_states[user_id] = {"state": "main_menu"}

        state = user_states[user_id]["state"]

        if content == "помощь":
            user_states[user_id] = {"state": "main_menu"}
            self.send_reply(message, main_menu)
            return

        if state == "main_menu":
            if re.match(r"^\d+$", content.strip()) and int(content.strip()) in all_theme_ids:
                user_states[user_id] = {"state": "select_level", "topic": int(content)}
                self.send_reply(message, "Какой у вас уровень знаний по теме? Выбери, введя соответствующую цифру: **`1. начальный`** / **`2. средний`** / **`3. продвинутый`**.\n\nЧтобы вернуться в главное меню, введи: **`помощь`**")
            else:
                self.send_reply(message, main_menu)

        elif state == "select_level":
            if content in ["1", "2", "3"]:
                user_states[user_id] = {"state": "chat", "topic": user_states[user_id]["topic"], "level": content}
                #open file
                #prompt model to summorize it
                topic = user_states[user_id]["topic"]

                file_path_summary = base_dir + 'course_materials/' + str(topic) + ".txt"

                with open(file_path_summary, 'r', encoding="utf-8") as sum_file:
                    sum_lines = sum_file.readlines()
                
                sum_firstlines = sum_lines[0]

                if sum_firstlines[0] == "#":
                    sum_firstlines = sum_firstlines[1:].strip()

                sum_restlines = " ".join(sum_lines[1:])

                summary_prompt = f"Сделай краткое резюме данного текста максимум на ДВА предложения, НЕ БОЛЬШЕ: {sum_restlines}"

                self.send_reply(message, "🕥 Подожди, я ищу свой конспектик по теме...")

                llm_summary = format_llm_prompt(summary_prompt)

                self.send_reply(message, f"**Тема: **{sum_firstlines}\n**Резюме по теме: **\n---\n{llm_summary}\n---\n**Давай начнём изучение темы, расскажи, что ты слышал по этой теме?**\n\nДля возврата в меню введите **`помощь`**.")
            else:
                self.send_reply(message, "Пожалуйста, выберите уровень, введя соответствующую цифру: **`1. начальный`**, **`2. средний`** или **`3. продвинутый`**.\n\nЧтобы вернуться в главное меню, введи: **`помощь`**")

        elif state == "chat":
            topic = user_states[user_id]["topic"]
            level = user_states[user_id]["level"]
            topic_name = lines[int(str(topic).strip())-1]
            
            model_level_desc = ''
            if level == 'начальный':
                model_level_desc = 'объяснения попроще для студентов с начальным уровнем знаний'
            if level == 'средний':
                model_level_desc = 'средний уровень сложности лексики - для студентов среднего уровня'
            if level == 'продвинутый':
                model_level_desc = 'сложный уровень лексики научный академический формальный язык - для студентов с продвинутым уровнем'

            #Main answer

            chunks_for_llm = search_chunks(content)

            user_query = f"""# Вводные данные:
- Ты преподователь-профессор в Высшей школе экономики и к тебе обращаются студенты по теме курса "{topic_name}".
- Ты должен ответить на вопрос студенту или просто поддрежать беседу по теме курса, если вопроса нет.
- У студента указан уровень знаний, используй {model_level_desc}.
- ПОМНИ, что нужно обязательно использовать контекст и информацию, предоставленную тебе по курсу.
- Длина твоего ответа и текста не должны быть больше 40 (сорок) слов.
# Структура ответа:
- Ответь на вопрос студента (в рамках курса) или поддержки разговор со студентом (в рамках курса) самым лучшим образом, последовательно и логично.
- Твой ответ НЕ ДОЛЖЕН ПРЕВЫШАТЬ по объему 40 (сорок) слов.
# Контекст для ответа:
- Вопрос студента: {content}.
- Контекст и информация по курсу: {chunks_for_llm}.
"""

            chunks_get = search_chunks_biblio(content)

            self.send_reply(message, "💭...")

            llm_main_response = format_llm_prompt(user_query)

            response = f"""**Debug info:** вы выбрали тему {topic}. {topic_name}. C уровнем знаний: {level}. Ваш вопрос: {content}\n
{llm_main_response}\n
\n```spoiler  📚 Источники информации:
{chunks_get}
```
"""

            self.send_reply(message, response)

            

            #Agents start

            self.send_reply(message, "✅ Подключаем агентов по проверке ответов...")

            agents_prompt = f"""<text_to_check>
{llm_main_response}
</text_to_check>

<information_text_must_be_based_on>
{chunks_for_llm}
</information_text_must_be_based_on>

Необходимо проверить текст, заключенный между тегами <text_to_check> и </text_to_check> на его соответсвие информации, заключенной между тегами <information_text_must_be_based_on> и </information_text_must_be_based_on>, проверить, что текст был написан с опорой исключительно на данную информацию. Если текст не соответсвует заданной информации, то ты должен исправить текст так, чтобы в нем была использована ТОЛЬКО предоставленная информация и НИКАКОЙ ДРУГОЙ ВНЕШНЕЙ ИНФОРМАЦИИ.
Структура ответа должна быть следующей: 1) оценка соответствия текста и информации, на основании которой он был написан; 2) исправленный текст с опорой только на данную информацию.
Текст является ответом на следующий вопрос: {content}.
"""

            self.send_reply(message, format_llm_prompt(agents_prompt))

            #Question back

            self.send_reply(message, "📝 Формирую план обсуждения...")

            new_know_chunks = search_chunks_new_knowledge(content)

            new_query = f"""Сформулируй один открытый вопрос по данной тебе информации, в ответе укажи только сам вопрос. Не обязательно пытаться использовать все факты из информации, можно ограничиться лишь частью информации и по ней сформулировать один открытый вопрос. Вот информация: {new_know_chunks}"""

            self.send_reply(message, f"""{format_llm_prompt(new_query)}""")

            self.send_reply(message, f"""
\n```spoiler  ⛔ Debug info:
{new_query}
```
""")



handler_class = QuizBot