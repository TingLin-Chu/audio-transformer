import os
import ffmpeg

mp4_folder = "F:\\XD_Violence\\Videos\\Training"
wav_folder = "F:\\XD_Violence\\Audios\\Training"

if not os.path.exists(wav_folder):
    os.makedirs(wav_folder)

for file in os.listdir(mp4_folder):
    if file.endswith(".mp4"):
        mp4_path = os.path.join(mp4_folder, file)
        wav_path = os.path.join(wav_folder, file.replace(".mp4", ".wav"))
        if os.path.exists(wav_path):
            print(f"Already converted video to .wav: {file}")
            continue
        print(f"Converting {file}...")
        ffmpeg.input(mp4_path).output(wav_path, acodec="pcm_s16le", ar="44100").run()

print(f"Successfully converted {mp4_folder} to {wav_folder}!")