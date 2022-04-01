import os
import imageio
## GF——ROI
# dir_path = r'F:\桌面\Multitemporalresult\classmap\SSRN\GF'
# save_dir = r'F:\桌面\Multitemporalresult\classmap\SSRN\GF_Region1'
# img_lists = os.listdir(dir_path)
#
# for img in img_lists:
#     if img[-3:] == 'png':
#         image = imageio.imread(os.path.join(dir_path, img))
#
#         sub_image = image[134:218, 166:250,:] # x1=166; y1=134; width1=85; height1=85;
#
#         save_path = os.path.join(save_dir, img)
#         imageio.imsave(save_path, sub_image)
#         print(img)

# PA——ROI——Region1
dir_path = r'F:\桌面\Multitemporalresult\classmap\SSRN\PA'
save_dir = r'F:\桌面\Multitemporalresult\classmap\SSRN\PA_Region1'
img_lists = os.listdir(dir_path)

for img in img_lists:
    if img[-3:] == 'png':
        image = imageio.imread(os.path.join(dir_path, img))

        sub_image = image[34:93, 1:65,:] # x1=1; y1=34; width1=63; height1=58;
        save_path = os.path.join(save_dir, img)
        imageio.imsave(save_path, sub_image)
        print(img)
# PA——ROI——Region2
dir_path = r'F:\桌面\Multitemporalresult\classmap\SSRN\PA'
save_dir = r'F:\桌面\Multitemporalresult\classmap\SSRN\PA_Region2'
img_lists = os.listdir(dir_path)

for img in img_lists:
    if img[-3:] == 'png':
        image = imageio.imread(os.path.join(dir_path, img))

        sub_image = image[236:301, 94:158,:] # x2=94; y2=236; width2=63; height2=64;
        save_path = os.path.join(save_dir, img)
        imageio.imsave(save_path, sub_image)
        print(img)

