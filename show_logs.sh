#!/bin/bash

for i in $(seq 0 7); do
    FILE="log_rank${i}.txt"

    echo "##########################################################"
    echo "### [START] CONTENT OF: $FILE"
    echo "##########################################################"

    if [ -f "$FILE" ]; then
        cat "$FILE"
    else
        echo "[Warning] File '$FILE' does not exist."
    fi

    echo ""
    echo "### [END] $FILE"
    echo -e "\n\n" 
done
