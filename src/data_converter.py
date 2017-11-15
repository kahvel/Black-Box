# from tensorflow.python.summary import event_accumulator
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import sys
import entropy_estimators

def create_csv(inpath, outpath):
    sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
          event_accumulator.IMAGES: 1,
          event_accumulator.AUDIO: 1,
          event_accumulator.SCALARS: 0,
          event_accumulator.HISTOGRAMS: 10000}
    ea = event_accumulator.EventAccumulator(inpath, size_guidance=sg)
    ea.Reload()
    histogram_tags = ea.Tags()['histograms']
    for tag in histogram_tags:
        events = ea.Histograms(tag)
        print(tag, events)
    # scalar_tags = ea.Tags()['scalars'] # scalars, histograms, distributions
    # df = pd.DataFrame(columns=scalar_tags)
    # for tag in scalar_tags:
    #     events = ea.Scalars(tag)
    #     scalars = np.array(map(lambda x: x.value, events))
    #     df.loc[:, tag] = scalars
    # df.to_csv(outpath)


if __name__ == '__main__':
    args = sys.argv
    inpath = "C:\\Users\\Anti\\PycharmProjects\\Black-Box\\src\\logs\\events.out.tfevents.1509623527.DESKTOP-N6N9NSM"
    outpath = "C:\\Users\\Anti\\PycharmProjects\\Black-Box\\src\\logs\\out"
    create_csv(inpath, outpath)
