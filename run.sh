#!/bin/bash
set -euo pipefail

rm -rf for_video
rm *.pth
mkdir for_video
mkdir for_video/saved_models
mkdir for_video/graphs

python main_video.py

# for i in {1..80}; do
#     python main_video.py

# 	echo "Run $i completed"

# done



