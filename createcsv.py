import os
import librosa
import numpy as np
import csv

script_directory = os.path.dirname(os.path.abspath(__file__)) #on recupere le chemin absolu d'ou se trouve notre fichier Python 
files = os.listdir(script_directory)#on liste les fichiers dans le repertoire

csv_header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    csv_header += ' mfcc' + str(i)
csv_header += ' spectral_contrast spectral_flatness tonnetz estimated_tempo genre/label'

with open('data.csv', 'w', newline='') as new_file:
    writer = csv.writer(new_file)
    writer.writerow(csv_header.split())

for file in files:
    
    if(os.path.isdir(file)):#on verifie que c'est bien un dossier
        files_in_directory = os.listdir(script_directory + '/' + file)#on recuperer l'ensemble des fichiers dans le dossier
        for file_in_directory in files_in_directory:
            print(file_in_directory)#pour evaluer l'avancement
            if(os.path.splitext(file_in_directory)[1] == '.wav'):#on verifie que l'extension des fichiers est bien en .wav
                file_name = os.path.splitext(file_in_directory)[0]#on recupere le nom du fichier
                
                genre = file_name.split('.')[0]#on recupere le genre
                
                #extraction des features
                y,sr = librosa.load(script_directory + '/' + file + '/' + file_in_directory, mono=True, duration=30)
                
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                rms = librosa.feature.rms(y=y)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                
                to_append = file_name + ' ' + str(np.mean(chroma_stft)) + ' ' + str(np.mean(rms)) + ' ' + str(np.mean(spec_cent)) + ' ' + str(np.mean(spec_bw)) + ' ' + str(np.mean(rolloff)) + ' ' + str(np.mean(zcr))
                for value in  mfcc:
                    to_append += ' ' + str(np.mean(value))
                    
                spectral_contrast = librosa.feature.spectral_contrast(y=y)
                spectral_flatness = librosa.feature.spectral_flatness(y=y)
                tonnetz = librosa.feature.tonnetz(y=y)
                
                tempo = librosa.beat.tempo(y=y)
                
                to_append += ' ' + str(np.mean(spectral_contrast)) + ' ' + str(np.mean(spectral_flatness)) + ' ' + str(np.mean(tonnetz)) + ' ' + str(tempo)[1:-1] + ' ' + genre
                
                with open('data.csv', 'a', newline='') as writing_new_file:
                    writer = csv.writer(writing_new_file)
                    writer.writerow(to_append.split())
                
print('Done !')


    