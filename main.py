#!usr/bin/python3
import json

# import watchdog utils
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import PatternMatchingEventHandler
import sys
import bot
import network
import utils 
import time 
import numpy as np 


def read_json(path):
    with open(path) as file:
        data = json.load(file)

    return data 
class GameStateHandler(PatternMatchingEventHandler):
    patterns = [sys.argv[1] + '/game_status.json']

    def __init__(self, bot_id):
        super().__init__()
        self.bot_id = bot_id 
        self.agent = network.Agent(
            gamma=0.99,
            epsilon=0.01,
            lr=0.0001,
            batch_size=64,
            n_actions=(len(np.unique(utils.ALL_LABELS))),
            eps_end=0.01, 
            input_dims=[utils.FEATURES_DIM],
            training_mode=False
        )


    def process(self, event):
        print('Game state changed: {} - {}'.format(event.src_path, event.event_type))
        gamestatus = read_json(sys.argv[1] + '/game_status.json')
        bot.play(self.bot_id, self.agent, gamestatus)


    def on_modified(self, event):
        self.process(event)


    def on_created(self, event):
        self.process(event)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        print("$ main.py <pasta io> <bot_id>")
        sys.exit()

    observer = Observer()
    observer.event_queue.maxsize = 0
    observer.schedule(GameStateHandler(int(args[1])), path=args[0] if args else '.')

    # watch for changes in game_status.json
    print('watching {} ...'.format(args[0]))
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()