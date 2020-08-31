import numpy as np
import math
import os
import cv2
import time
import librosa
import librosa.display
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips, CompositeVideoClip
from skimage.filters import threshold_yen
import pandas
from skimage import io
import urllib

video_file_path_train = './data/train/video/'
audio_file_path_train = './data/train/audio/'
spec_file_path_train = './data/train/spec/'

video_file_path_test = './data/test/video/'
audio_file_path_test = './data/test/audio/'
spec_file_path_test = './data/test/spec/'

video_link = []
main_speaker = ['SirKenRobinson', 'AlGore', 'DavidPogue', 'MajoraCarter', 'HansRosling', 'TonyRobbins', 'JuliaSweeney', 'JoshuaPrinceRamus', 'DanDennett', 'RickWarren']
year = ['2006', '2006','2006', '2006', '2006', '2006', '2006','2006', '2006', '2006']

transcripts = []
speech_recog = []
speech_recog_ref = []

IMG_HEIGHT_MFCC = 20
IMG_HEIGHT_MEL = 128
IMG_WIDTH = 44

duration = 1

def downsize_videos(mp4_path, des_path):

    # Downsize video to only 1 minute
    starting_point = 60
    ending_point = 120

    clip = VideoFileClip(mp4_path)
    subclip = clip.subclip(starting_point, ending_point)
    subclip.write_videofile(des_path)

def merge_videos(path_source_1, path_source_2, mixed_path):

    clip01 = VideoFileClip(path_source_1)
    clip02 = VideoFileClip(path_source_2)

    clip01 = clip01.resize(0.60)
    clip02 = clip02.resize(0.60)

    final_clip = CompositeVideoClip([clip01.set_position(("left","center")), clip02.set_position(("right","center"))], size=(720, 460))
    final_clip.write_videofile(mixed_path)
    
def download_video():

    for i, (each_main_speaker, each_year) in enumerate(zip(main_speaker, year)):
        mp4url = 'https://download.ted.com/talks/' + each_main_speaker + '_' + each_year + '-480p.mp4' + '?apikey=acme-roadrunner'
        urllib.request.urlretrieve(mp4url, video_file_path_train + 'raw/' + each_main_speaker + '_' + each_year + '.mp4')

def mp4_to_wav(mp4_path, wav_path):

    command = 'ffmpeg -i ' + mp4_path + ' -ab 160k -ac 2 -ar 44100 -vn ' + wav_path
    os.system(command)

def main():

    download_video()

    for each_main_speaker, each_year in zip(main_speaker, year):
        downsize_videos(mp4_path=video_file_path_train + 'raw/' + each_main_speaker + '_' + each_year + '.mp4', des_path=video_file_path_train + 'sub/' + 'sub_' + each_main_speaker + '_' + each_year + '.mp4')
        mp4_to_wav(mp4_path=video_file_path_train + 'sub/' + 'sub' + '_' + each_main_speaker + '_' + each_year + '.mp4', wav_path=audio_file_path_train + each_main_speaker + '_' + each_year + '.wav')

    for each_main_speaker_target, each_year_target in zip(main_speaker, year):

        for each_main_speaker_mixed, each_year_mixed in zip(main_speaker, year):

            if each_main_speaker_mixed == each_main_speaker_target:
                continue
            else:
                merge_videos(path_source_1=video_file_path_train + 'sub/' + 'sub_' + each_main_speaker_target + '_' + each_year_target + '.mp4', path_source_2=video_file_path_train + 'sub/' + 'sub_' + each_main_speaker_mixed + '_' + each_year_mixed + '.mp4', mixed_path=video_file_path_train + 'mixed/' + 'mixed' + '_' + each_main_speaker_target + '_' + each_year_target + '_' +  each_main_speaker_mixed + '_' + each_year_mixed + '.mp4')
                mp4_to_wav(mp4_path=video_file_path_train + 'mixed/' + 'mixed' + '_' + each_main_speaker_target + '_' + each_year_target + '_' +  each_main_speaker_mixed + '_' + each_year_mixed + '.mp4', wav_path=audio_file_path_test + 'mixed/' + 'mixed' + '_' + each_main_speaker_target + '_' + each_year_target + '_' +  each_main_speaker_mixed + '_' + each_year_mixed + '.wav')

if __name__ == "__main__":
    main()