from fastapi import UploadFile
from typing import List


async def get_contents_from_files(files: List[UploadFile]) -> List[bytes]:
    """
    Given a list of files, assume the first one contains the main face and the rest
    are pictures to check for the first face.
    :param files: List of files
    :return: the main image and a list of images to check
    """
    return [await file.read() for file in files]