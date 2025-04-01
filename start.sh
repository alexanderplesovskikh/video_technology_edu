#!/bin/bash
cd ../../../../../python-zulip-api
python3 ./tools/provision
source ./zulip-api-py3-venv/bin/activate

cd zulip_bots/zulip_bots/bots/videoedu/
docker compose build && docker compose up -d
nohup python3 worker_videoedu.py > zulip_worker.log 2>&1 &
nohup zulip-run-bot videoedu --config-file zuliprc > zulip_bot.log 2>&1 &
