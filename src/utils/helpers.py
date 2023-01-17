import winsound


def tone(duration=3, note='a'):
    """
    Plays a tone for the number of seconds specified
    :param duration: seconds to play tone
    :param note: chose the lowercase letter that corresponds to a musical note a-g.
    :return: None
    """
    musical_dict = {'a': 440, 'b': 494, 'c': 523, 'd': 587, 'e': 659, 'f': 698, 'g': 784}
    duration_ms = duration * 1000  # convert seconds to milliseconds

    winsound.Beep(musical_dict[f'{note}'], duration_ms)  # play tone


def combos(
        low_res_list: list[int],
        high_res_list: list[int],
        scale_factors: list[int] = (2, 4, 8)
) -> list[tuple[int, int]]:
    """
    Generate size combination tuples. Used for the SRGAN image sizes themselves.

    :return: List of tuples ordered: (low, high)
    """
    return [
        (lr, hr)
        for lr in low_res_list
        for hr in high_res_list
        if (hr // lr) in scale_factors
    ]
