#!/bin/bash
youtube-dl -f 134 'https://www.youtube.com/watch?v=Ww2r3FcsiTY'

mkdir crops_0+
mkdir crops_5040+

ffmpeg -i *.mp4 -t 5040 -r 1 -filter:v "crop=80:48:550:20" crops_0+/out_%08d.png

ffmpeg -ss 01:20:00 -i *.mp4 -ss 240 -r 1 -filter:v "crop=80:48:560:65" crops_5040+/out_%08d.png

mkdir templates
echo "Now manually crop out some templates..."

