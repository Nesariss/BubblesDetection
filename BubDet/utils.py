import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
import exifread
#from app import get_exif_data

def decode_image(img_str):
    """Decodes an image from a b64 image string.
    Args:
        img_str (str): An image represented as a b64 string.
    Returns:
        ndarray: An image represented as a ndarray (opencv format).
    """
    decoded = base64.b64decode(img_str.split("base64,")[-1])
    image = Image.open(BytesIO(decoded))
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def encode_image(image):
    """Encodes an image to b64 string format.
    Args:
        image (ndarray): An image in array format (can be read by opencv library).
    Returns:
        str: Data representing an image in a b64 string format.
    """
    # convert opencv image to PIL
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    buffered = BytesIO()
    im_pil.save(buffered, format="JPEG")
    b64_string = base64.b64encode(buffered.getvalue())
    b64_string = "data:image/jpeg;base64," + b64_string.decode()
    return b64_string

def get_exif_data(image_bytes):
    """Extracts EXIF data from an image.
    Args:
        image_bytes (bytes): An image represented as a bytes object.
    Returns:
        dict: A dictionary containing the extracted EXIF data.
    """
    exif_data = {}
    with BytesIO(image_bytes) as f:
        tags = exifread.process_file(f)
        for tag, value in tags.items():
            exif_data[tag] = str(value)
    
    #print(exif_data)
    return exif_data


def get_meta_info(img_str):

    img_bytes = base64.b64decode(img_str.split("base64,")[-1])

    # Extract EXIF data
    exif_data = get_exif_data(img_bytes)

    # Print EXIF data
    print("Aperture:", exif_data.get("EXIF ApertureValue"))
    print("Exposure Bias:", exif_data.get("EXIF ExposureBiasValue"))
    print("Focal Length:", exif_data.get("EXIF FocalLength"))
    FocalLength = exif_data.get("EXIF FocalLength")

    return FocalLength