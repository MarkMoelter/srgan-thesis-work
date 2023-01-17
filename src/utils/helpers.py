import os

import winsound


def create_path(dir_path: str) -> None:
    """
    Create a folder at the path provided
     if it does not already exist at that path.
    
    :param dir_path: Path to create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


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


def img_save(
        images,
        save_path,
        desc='[saving results]',
        grid=False,
        nrow=3,
        position=0,
        leave=True
):
    """Saves a list of images using\n
    torch.utils.save_image()\n,
    creates save_path dir.
    """
    from tqdm import tqdm
    import torchvision.utils as utils

    create_path(save_path)
    for index, image in enumerate(tqdm(images, desc=desc, position=position, leave=leave)):
        zfill_idx = str(index).zfill(4)

        if grid:
            image = utils.make_grid(image, nrow=nrow)  # number of images in each row
        img_out = f'{save_path}/{zfill_idx}.png'
        utils.save_image(image, img_out)


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
