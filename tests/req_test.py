# import cv2
#
# from utils.external import handlers
#
# # handlers.install_dependencies(notebook=True)

from utils.data import cityscapes


username = '...'
password = '...'

batch_size = {'train': 1000}

cityscapes.Info.cache_dir = './cache/cityscapes/panoptic_parts'

info = cityscapes.Info.panoptic_parts

# cv_decode = True, runs up to 1.8x faster
cityscapes.get(username, password, info, dest='./',
               shape=(256, 128), batch_size=batch_size,
               dname='left_view', prefix='data',
               cv_decode=True)
