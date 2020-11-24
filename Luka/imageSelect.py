import easygui

selectedFiles = easygui.fileopenbox("You can choose multiple", "Image Select", filetypes=["*.png", "*.jpg"], multiple=True)

for img in selectedFiles:
    print(img)
    # read image
    # call the algorithm
