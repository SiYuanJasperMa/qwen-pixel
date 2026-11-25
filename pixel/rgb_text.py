"""Lossless, human-readable RGB text. No image tokenizer or learned codec."""
import numpy as np
from PIL import Image


def encode_rgb(image, encoding='hex_rle'):
    if encoding not in ('decimal', 'hex', 'hex_rle'):
        raise ValueError('Unsupported RGB encoding: ' + encoding)
    a = np.asarray(image.convert('RGB'), dtype=np.uint8)
    h, w = a.shape[:2]
    lines = [f'RGB_TEXT_V1 width={w} height={h} channels=RGB uint8 range=0..255 order=row-major',
             'Each row y starts at x=0. Hex rrggbb denotes R,G,B; rrggbb*N repeats that pixel N times.'
             if encoding.startswith('hex') else 'Each row y starts at x=0; pixels are decimal R,G,B triples.',
             f'encoding={encoding}']
    for y, row in enumerate(a):
        if encoding == 'decimal':
            pixels = ['%d,%d,%d' % tuple(p) for p in row]
        else:
            colors = ['%02x%02x%02x' % tuple(p) for p in row]
            pixels = []
            i = 0
            while i < w:
                j = i + 1
                if encoding == 'hex_rle':
                    while j < w and colors[j] == colors[i]: j += 1
                pixels.append(colors[i] + (f'*{j-i}' if j-i > 1 else ''))
                i = j
        lines.append(f'y={y}: ' + ' '.join(pixels))
    return '\n'.join(lines)


def decode_rgb(text):
    header = text.splitlines()[0]
    fields = dict(part.split('=',1) for part in header.split() if '=' in part)
    w, h = int(fields['width']), int(fields['height'])
    encoding = text.splitlines()[2].split('=',1)[1]
    if w < 1 or h < 1 or encoding not in ('decimal', 'hex', 'hex_rle'):
        raise ValueError('Invalid dimensions or encoding')
    pixels = []
    for y, line in enumerate(text.splitlines()[3:]):
        prefix, values = line.split(': ',1)
        if prefix != f'y={y}': raise ValueError('Non-contiguous rows')
        row = []
        for value in values.split():
            if encoding == 'decimal':
                color = tuple(map(int, value.split(','))); count = 1
            else:
                parts = value.split('*'); rgb = parts[0]
                if len(rgb) != 6 or len(parts) > 2:
                    raise ValueError('Invalid hex pixel')
                color = tuple(int(rgb[i:i+2],16) for i in (0,2,4))
                count = int(parts[1]) if len(parts)>1 else 1
            if len(color)!=3 or any(v<0 or v>255 for v in color) or count<1:
                raise ValueError('Invalid RGB pixel')
            if len(row) + count > w:
                raise ValueError('Row exceeds declared width')
            row.extend([color]*count)
        if len(row)!=w: raise ValueError('Row width mismatch')
        pixels.append(row)
    if len(pixels)!=h: raise ValueError('Height mismatch')
    return Image.fromarray(np.asarray(pixels,dtype=np.uint8))
