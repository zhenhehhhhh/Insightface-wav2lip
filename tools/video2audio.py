import subprocess

command ="ffmpeg -i cai2.mp4 -ab 160k -ac 2 -ar 44100 -vn cai.wav"

subprocess.call(command, shell=True)