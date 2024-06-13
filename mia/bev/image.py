from PIL import Image


def center_pad(img: Image, width: int, height: int):
    if img.width < width or img.height < height:
        height = max(img.height, height)
        width = max(img.width, width)
        padded_img = Image.new("RGBA", (width, height), (0, 0, 0, 0)) 
        x_offset = (width - img.width) // 2
        y_offset = (height - img.height) // 2
        padded_img.paste(img, (x_offset, y_offset))
        img = padded_img
    return img

def center_crop_to_size(img: Image, width: int, height: int) -> Image:
    """Center crop the image to the given width and height."""
    if img.width < width: 
        raise ValueError("Invalid crop width. Crop width is larger than image width.")
    if img.height < height:
        raise ValueError("Invalid crop height. Crop height is larger than image height.")
    left = (img.width - width) / 2
    top = (img.height - height) / 2
    right = (img.width + width) / 2
    bottom = (img.height + height) / 2
    img = img.crop((left, top, right, bottom))
    return img