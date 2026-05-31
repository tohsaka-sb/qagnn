import os
import numpy as np
from numpy.lib import format


SOURCE_PATH = 'data/cpnet/tzw.ent.npy'
TARGET_PATH = 'data/cpnet/tzw.ent.repaired.npy'


def read_npy_header(path):
    with open(path, 'rb') as fin:
        major, minor = format.read_magic(fin)
        shape, fortran_order, dtype = format._read_array_header(fin, version=(major, minor))
        header_end = fin.tell()
    return shape, fortran_order, dtype, header_end


def main():
    shape, fortran_order, dtype, header_end = read_npy_header(SOURCE_PATH)
    if fortran_order:
        raise ValueError('Fortran-order arrays are not supported by this repair script.')

    row_width = shape[1]
    itemsize = np.dtype(dtype).itemsize
    payload_bytes = os.path.getsize(SOURCE_PATH) - header_end
    recovered_rows = payload_bytes // (row_width * itemsize)
    expected_rows = shape[0]

    if recovered_rows <= 0:
        raise ValueError('No complete rows could be recovered from the truncated embedding file.')

    print(f'source shape in header: {shape}, dtype={dtype}')
    print(f'file contains {recovered_rows}/{expected_rows} complete rows; padding the remainder with zeros')

    repaired = format.open_memmap(TARGET_PATH, mode='w+', dtype=dtype, shape=shape)
    repaired[:] = 0

    partial = np.fromfile(SOURCE_PATH, dtype=dtype, offset=header_end, count=recovered_rows * row_width)
    repaired[:recovered_rows] = partial.reshape(recovered_rows, row_width)
    repaired.flush()

    print(f'repaired embeddings written to {TARGET_PATH}')


if __name__ == '__main__':
    main()
