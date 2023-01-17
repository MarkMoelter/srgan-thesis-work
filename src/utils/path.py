def create_path(dir_path: str) -> None:
    """
    Create a folder at the path provided
     if it does not already exist at that path.

    :param dir_path: Path to create.
    """
    import os

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


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
