import os

# Brightness
old_basepath = 'Brightness_Adjustment/'
basepath = 'Brightness/'
os.rename(old_basepath, basepath)

for filename in os.listdir(basepath):
    filename = basepath + filename
    new_filename = filename.replace('+', '')
    os.rename(filename, new_filename)

# Contrast
old_basepath = 'Contrast_Adjustment/'
basepath = 'Contrast/'
os.rename(old_basepath, basepath)

for filename in os.listdir(basepath):
    filename = basepath + filename
    new_filename = filename.replace('+', '')
    os.rename(filename, new_filename)


# Gamma
old_basepath = 'Gamma_Correction/'
basepath = 'Gamma/'
os.rename(old_basepath, basepath)


# Image_Rotation
old_basepath = 'Image_Rotation/'
basepath = 'Rotation/'
os.rename(old_basepath, basepath)
for filename in os.listdir(basepath):
    filename = basepath + filename
    new_filename = filename.replace('+', '')
    os.rename(filename, new_filename)


# Image_Scaling
old_basepath = 'Image_Scaling/'
basepath = 'Image_Scaling/'
os.rename(old_basepath, basepath)
for filename in os.listdir(basepath):
    filename = basepath + filename
    new_filename = filename.replace('scaling', 'image_scaling')
    os.rename(filename, new_filename)


# JPEG
old_basepath = 'JPEG_Compress/'
basepath = 'JPEG/'
os.rename(old_basepath, basepath)
for filename in os.listdir(basepath):
    filename = basepath + filename
    new_filename = filename.replace('compressQuality', 'jpeg')
    os.rename(filename, new_filename)


# JPEG
old_basepath = 'JPEG_Compress/'
basepath = 'JPEG/'
os.rename(old_basepath, basepath)
for filename in os.listdir(basepath):
    filename = basepath + filename
    new_filename = filename.replace('compressQuality', 'jpeg')
    os.rename(filename, new_filename)


# SaltAndPepperNoise
old_basepath = 'SaltAndPepperNoise/'
basepath = 'Salt_and_pepper_noise/'
os.rename(old_basepath, basepath)
for filename in os.listdir(basepath):
    filename = basepath + filename
    new_filename = filename.replace('SaltAndPepperNoise_ratio', 'salt_and_pepper_noise')
    os.rename(filename, new_filename)


