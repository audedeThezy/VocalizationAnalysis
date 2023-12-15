import os
import datetime
import logging
import sys

from avisoft.functions_avisoft import *
from blocks.FED3.Fed3Manager3 import Fed3Manager3
from time import sleep

sys.path.append( "C:\\Users\\de Witasse Thézy\\Documents\\ENS\\cours_ENS\\M2\\MoBi\\projet" )
from RecordingAnalysis3 import VocAnalysis

from datetime import datetime

############

method = "VocNumber"
thresholds = [1, 100]

###########

class Experiment(object):

    def __init__(self):

        self.fed = Fed3Manager3( comPort="COM3" , name= "Fed")
        self.fed.addDeviceListener(self.listener)

        # iniate the playback as enable
        self.enable_playback = True

        # initiate the feeding as enable
        self.enable_feeding = True

        # initiate the recording as enable
        self.enable_recording = True

        # iniate the trial id to 1
        self.trial = 1

        # path of CMCDDE.EXE for avisoft
        self.path_cmcdde = "C:/Users/adminlocal/Documents/conditioning_sm/RECORDER_DDE/CMCDDE.EXE"

        # setup the id of df file
        self.df_id = f"C:/Users/adminlocal/Documents/GitHub/go-nogo/experiments/pilots/df/df_{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

        # immediatly write the headers of df file
        with open(f"{self.df_id}", 'w') as self.df_file:
            print(f"trial;datetime;event;data", file = self.df_file)
        self.df_file.close()

        # setup the wav file to send
        self.wav = "C:/Users/adminlocal/Documents/conditioning_sm/sounds/go/WN_1s.wav"

        # start the recording and add to the logs
        logging.info("recording start")
        start_recording(self.path_cmcdde)



    def listener(self, event):

        self.ev = event.description
        self.dt = event.datetime
        self.d = event.data

        # print(self.dt)
        # print(self.ev)
        # print(self.d)
        # print("\n")

        if "nose poke" in event.description:

            logging.info("nose poke")

            with open(f"{self.df_id}", 'a') as self.df_file:
                print(f"{self.trial};{self.dt};{self.ev};{self.d}", file = self.df_file)
            self.df_file.close()

            # play a sounfile and write it as an event
            if self.enable_playback == True:

                logging.info('Playing wav file...')

                play(cmcdde = self.path_cmcdde, wav_file = self.wav)

                with open(f"{self.df_id}", 'a') as self.df_file:
                    playback_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                    print(f"{self.trial};{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')};playback;{self.wav}", file = self.df_file)
                self.df_file.close()

                self.enable_playback = False

                sleep(0.05)  # TODO : check if sleep is long enough

            # TODO: give pellet if voc
            if VocAnalysis(playback_time_str, basepath, method, thresholds) == False :
                """
                method = VocNumber, VocDuration, VocFrequecy
                thresholds = interval of numbers, durations, frrequencies
                False if thersholds conditions unssatisfied.
                """
                self.enable_feeding == False

            # feed the animal and write it as an event
            if self.enable_feeding == True:
                # condition on recording
                # if len(ExtractRecording()) > 0
                # logging.info("%s vocalizations detected" % len(ExtractRecording()))
                self.fed.feed()

                logging.info("feeding")

                with open(f"{self.df_id}", 'a') as self.df_file:
                    print(f"{self.trial};{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')};feeding;None", file = self.df_file)
                self.df_file.close()
                self.enable_feeding = False

        if "pellet delivered" in event.description:

            logging.info("pellet delivered")

            with open(f"{self.df_id}", 'a') as self.df_file:
                print(f"{self.trial};{self.dt};{self.ev};{self.d}", file = self.df_file)
            self.df_file.close()


        if "pellet picked" in event.description:

            # sleep(0.25)
            # self.fed.checkPellet()
            # sleep(0.25)

            logging.info("pellet picked")

            with open(f"{self.df_id}", 'a') as self.df_file:
                print(f"{self.trial};{self.dt};{self.ev};{self.d}", file = self.df_file)
            self.df_file.close()

            self.trial += 1
            self.enable_playback = True
            self.enable_feeding = True



        # if "pellet already delivered" in event.description:

        #     logging.info("pellet already delivered")



        # if "no pellet" in event.description:

        #     logging.info("no pellet")

#
#
#

# delay = ?


# Launch experiment

if __name__ == '__main__':

    logFile = f"C:/Users/adminlocal/Documents/GitHub/go-nogo/experiments/pilots/logs/logs_{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    print("Logfile: " , logFile )
    logging.basicConfig(level=logging.INFO, filename=logFile, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Application started')

    experiment = Experiment()

    while( True ):

        experiment.fed.read()
        sleep(0.01)

        if experiment.trial > 500:
            stop_recording(experiment.path_cmcdde)
            break

####### time test


