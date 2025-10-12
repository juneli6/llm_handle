#!/bin/bash

# 递归列出或杀掉给定进程的子进程
kill_or_list_process_tree() {
    local parent_pid=$1
    local list_only=$2

    # 找到所有子进程的PID
    local child_pids=$(pgrep -P $parent_pid)

    # 递归处理每个子进程
    for child_pid in $child_pids; do
        kill_or_list_process_tree $child_pid $list_only
    done

    # 列出或终止父进程
    if [ "$list_only" = true ]; then
        echo "进程ID: $parent_pid"
    else
        echo "杀掉进程ID: $parent_pid"
        kill -9 $parent_pid
    fi
}


# 正则表达式，匹配目标进程
PATTERN="deploy_vllm.sh"
# PATTERN="deploy_vllm.sh|vllm serve|-c from multiprocessing.resource_tracker import main|-c from multiprocessing.spawn import spawn_main"

if [ "$1" == "-l" ] || [ "$1" == "--list" ]; then
    LIST_ONLY=true
else
    LIST_ONLY=false
fi

# 查找匹配的父进程，并过滤掉grep本身的进程
PROCESS_LIST=$(ps aux | grep -E "$PATTERN" | grep -v grep)

# 检查是否有匹配的进程
if [ -z "$PROCESS_LIST" ]; then
    echo "没有找到匹配的进程。"
else
    echo "找到以下匹配的进程："
    echo "$PROCESS_LIST"

    if [ "$LIST_ONLY" = true ]; then
        echo "仅查看模式下，将列出所有匹配的进程及其子进程。"
    fi

    # 提取父进程ID
    PIDS=$(echo "$PROCESS_LIST" | awk '{print $2}')

    # 递归列出或终止所有匹配的进程
    echo "处理以下进程及其子进程："
    for PID in $PIDS; do
        kill_or_list_process_tree $PID $LIST_ONLY
    done

    if [ "$LIST_ONLY" = true ]; then
        echo "仅查看模式下，未终止任何进程。"
    else
        echo "所有进程终止完成。"
    fi
fi
