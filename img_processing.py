from PIL import Image
import numpy as np

def main():
    img = Image.open('mona_lisa.jpg')
    gray_scale = img.convert('L')

    intensity_array_2d = np.array(gray_scale)
    print(intensity_array_2d)
main()
