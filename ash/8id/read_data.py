import numpy as np
import dxchange

u = np.load('/data/vnikitin/8id/Eiffel_on_Si_large.npy')
print(u.shape)
# dxchange.write_tiff(u.real.astype('float32'),'/data/vnikitin/8id/re.tiff')
# dxchange.write_tiff(u.imag.astype('float32'),'/data/vnikitin/8id/im.tiff')
# dxchange.write_tiff(u.astype(''))

# print(u[0,0,0])