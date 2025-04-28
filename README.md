# Educator bot (SVT)

Бот-обучатор. В ранней стадии.

## Запуск модели ollama

1. Скачать ollama по ссылке: https://ollama.com/download/
2. Выполнить команду

```bash
ollama run gemma3:4b
```

3. Проверить, что модель скачалась

```bash
ollama list
```

## Запуск бота

Клонируем репозиторий python-zulip-api

```bash
git clone https://github.com/zulip/python-zulip-api.git
```

Перемещаемся в папку bots

```bash
cd python-zulip-api/zulip_bots/zulip_bots/bots/
```

Здесь содержится несколько ботов, добавим сюда и нашего

```bash
git clone https://git.miem.hse.ru/301/video-technology-educator.git
```

Перейдём в созданную папку

```bash
cd video-technology-educator
```

Наконец, запустим бота с помощью скрипта

```bash
sudo ./start.sh
```
