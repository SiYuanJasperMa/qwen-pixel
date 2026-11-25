import numpy as np
import pytest
from PIL import Image

from pixel.rgb_text import encode_rgb, decode_rgb
from pixel.rgb_text_eval import pixel_metrics


@pytest.mark.parametrize('encoding', ['decimal', 'hex', 'hex_rle'])
def test_lossless_uint8_and_row_order(encoding):
    pixels = np.random.default_rng(23).integers(0, 256, (7, 13, 3), dtype=np.uint8)
    pixels[1, :8] = [0, 255, 128]
    pixels[2, 0] = [255, 0, 255]
    text = encode_rgb(Image.fromarray(pixels), encoding)
    assert np.array_equal(pixels, np.asarray(decode_rgb(text)))


def test_rle_rejects_declared_width_overflow():
    text = encode_rgb(Image.new('RGB', (5, 2), 'white'))
    with pytest.raises(ValueError, match='exceeds'):
        decode_rgb(text.replace('ffffff*5', 'ffffff*5000000', 1))


def test_malformed_hex_and_unknown_encoding_are_rejected():
    text = encode_rgb(Image.new('RGB', (1, 1), 'black'), 'hex')
    with pytest.raises(ValueError, match='hex'):
        decode_rgb(text.replace('000000', '000000ff'))
    with pytest.raises(ValueError, match='Unsupported'):
        encode_rgb(Image.new('RGB', (1, 1)), 'base64')


def test_aligned_metrics_accept_no_mask_and_distinguish_noop(tmp_path):
    source, target = tmp_path / 'source.png', tmp_path / 'target.png'
    Image.new('RGB', (5, 7), (0, 0, 0)).save(source)
    Image.new('RGB', (5, 7), (255, 0, 0)).save(target)
    noop = pixel_metrics(source, target, source)
    exact = pixel_metrics(source, target, target)
    assert noop['gain'] == 0 and noop['changed_pixel_fraction'] == 0
    assert exact['l1'] == 0 and exact['gain'] > 0 and exact['changed_pixel_fraction'] == 1
