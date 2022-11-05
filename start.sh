#!/bin/bash
env=$1
echo "env args is：$1"
processName='gunicorn'
PID=$(ps -elf |grep $processName|grep -v grep |head -n 1 |awk '{printf $4}')
if [ $? -eq 0 ] && [ ${#PID} -gt 0 ]; then
    echo "process id : $PID"
    kill  ${PID}
    if [ $? -eq 0 ]; then
        echo "kill $processName success"
    else
        echo "kill $processName fail"
        exit 1
    fi
else
    echo "process $processName not exist"
fi
nohup gunicorn -c gunicorn_conf.py run_api:app &> ./logs/uwsgi.log &
