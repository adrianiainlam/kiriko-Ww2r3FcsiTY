#!/bin/bash
youtube-dl --hls-use-mpegts -f 134 'https://www.youtube.com/watch?v=04la0TDB-Kg'

mkdir crops

ffmpeg -i 【ヨッシーアイランド】赤ちゃんの飲酒恐竜運転！！【Vtuber_かくきりこ】-04la0TDB-Kg.mp4 -r 1  -filter:v "crop=80:48:560:20"  crops/out_%08d.png

mkdir templates
echo "Now manually crop out some templates..."

