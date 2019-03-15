from PIL import Image
import subprocess
import io

def get_img():
    """
    will return None if no image in clipboard
    """
    try:
        result = subprocess.run(["xclip", "-se", "c", "-t", "image/png", "-o"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        return Image.open(io.BytesIO(result.stdout))
    except OSError:
        return None

get_img().save("wow.png")
