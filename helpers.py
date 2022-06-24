def create_pth(x: str):
    """Create a folder if it does not already exist at that path"""
    import os
    if not os.path.exists(x):
        os.makedirs(x)
    return x

def log_init(name=None, level='logging.DEBUG', filepath='logs'):
    """
    Initializes logger and names file using today's date.
    :param name: Name of logger
    :param level: Logging level
    :param filepath: Filepath to logs
    :return: logger object to use in functions
    """
    import logging
    import datetime as dt

    # collect date
    today = dt.datetime.today()
    log_filename = f'{today.month:02d}-{today.day:02d}-{today.year}.log'

    # formatted logger
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )
    # create file handler
    fh = logging.FileHandler(f'{filepath}/{log_filename}')
    fh.setFormatter(formatter)

    # create stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    # create logger object and set the level
    logger = logging.getLogger(name)
    logger.setLevel(eval(level))

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

def combo_gen(lr_list, sr_list, except_list):
    a = []
    for LR in lr_list:
        for SR in sr_list:
            if f'{LR}-{SR}' in except_list:
                continue
            else:
                a.append(f'{LR}-{SR}')

    return a

def tone(duration=3, note='a'):
    """
    Plays a tone for the number of seconds specified
    :param duration: seconds to play tone
    :param note: chose the lowercase letter that corresponds to a musical note a-g.
    :return: None
    """
    import winsound
    musical_dict = {'a': 440, 'b': 494, 'c': 523, 'd': 587, 'e': 659, 'f': 698, 'g': 784}
    duration_ms = duration * 1000  # convert seconds to milliseconds

    winsound.Beep(musical_dict[f'{note}'], duration_ms)  # play tone
