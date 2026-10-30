import os

# Some cameras write slightly broken H.264 headers; FFmpeg then prints a warning
# for every frame although decoding works. Must be set before OpenCV loads FFmpeg.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
