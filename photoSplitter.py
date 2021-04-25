from scanner import splitPhotos
import glob

photos = glob.glob(".\\resources\\*.jpg")

#debug
# 0-show nothing
# 1-show everything
# 2-show outputs

#write
# 0-do nothing
# 1-save output

#split photo, if okay save if not flag that photo and move on
for photo in photos:
    splitPhotos(photo, debug = 0, write =1)
