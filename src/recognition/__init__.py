import pandas as pd


def recognition(plate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses OCR to recognize the high resolution and low resolution plates.

    :param plate_df: The dataframe containing the plate labels.
    """
    import logging
    from .recognition import Recognition

    # collect the OCR predictions
    recog = Recognition(plate_df)
    recog.benchmark_recognition()
    recog.low_resolution_recognition()
    recog.high_resolution_recognition()

    logging.info('Plate predictions finished')
    return recog.df
