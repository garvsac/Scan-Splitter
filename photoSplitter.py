from scanner import splitPhotos
import glob

photos = glob.glob(".\\resources\\*")

#split photo, if okay save if not flag that photo and move on
for photo in photos:
    splitPhotos(photo)
