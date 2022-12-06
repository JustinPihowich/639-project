from PIL import Image

input_image = Image.open("blemish.jpg")

pixel_map = input_image.load()

width, height = input_image.size

for i in range(width):
    for j in range(height):
        r, g, b = input_image.getpixel((i,j))
        if g < 80 or b < 100 or g > 130 or g < 100:
            pixel_map[i, j] = (255, 255, 255)
input_image.save("grayscale.png", format="png")
