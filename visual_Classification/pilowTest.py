from PIL import Image, ImageFilter
import numpy as np


# Open an image file
original_image = Image.open("simba.png")

# Display basic information about the image
print("Image Format:", original_image.format)
print("Image Mode:", original_image.mode)
print("Image Size:", original_image.size)

# Show the original image
# original_image.show()

# Convert the image to grayscale
gray_image = original_image.convert("L")
# gray_image.show()

# Apply a blur filter to the original image
blurred_image = original_image.filter(ImageFilter.BLUR)
# blurred_image.show()

# Resize the image
resized_image = original_image.resize((300, 300))
# resized_image.show()

# Save the modified images
gray_image.save("gray_image.jpg")
blurred_image.save("blurred_image.jpg")
resized_image.save("resized_image.jpg")


# # Récupérer les valeurs de tous les pixels sous forme d'une matrice
# mat = np.array(original_image)

# # Afficher la taille de la matrice de pixels
# print("Taille de la matrice de pixels : {}".format(mat.shape))

# Get image size
width, height = original_image.size

# # Set the chunk size
# chunk_size = 10000  # Adjust this based on available memory and image size

# # Iterate over chunks and retrieve pixel values
# for i in range(0, width, chunk_size):
#     for j in range(0, height, chunk_size):
#         chunk = original_image.crop((i, j, i + chunk_size, j + chunk_size))
#         mat_chunk = np.array(chunk)
#         print("Processed chunk at position ({}, {})".format(i, j))

# Set the line size
line_size = 1000  # Adjust this based on available memory and image size

# Iterate over lines and retrieve pixel values
for j in range(0, height, line_size):
    box = (0, j, width, min(j + line_size, height))
    line = original_image.crop(box)
    mat_line = np.array(line)
    print("Processed line from {} to {}".format(j, min(j + line_size, height)))