"""Guard source-only execution without excluding normal editing programs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_magicbrush_target_informed import looks_hardcoded


def test_source_coordinates_and_mask_name_are_allowed():
    code = "def edit(image):\n    mask_image = image.copy()\n    mask_image.putpixel((10, 20), (0, 0, 255))\n    return mask_image\n"
    assert not looks_hardcoded(code)[0]


def test_program_length_is_not_target_leakage():
    code = 'def edit(image):\n' + ('    # source-dependent implementation detail\n' * 400) + '    return image.copy()\n'
    assert len(code) > 12000
    assert not looks_hardcoded(code)[0]


def test_external_image_read_is_rejected_without_a_literal_filename():
    assert looks_hardcoded('def edit(image):\n    return Image.open(secret_path)\n')[0]


def test_pixel_access_load_is_allowed():
    assert not looks_hardcoded('def edit(image):\n    pixels=image.load()\n    return image\n')[0]


def test_single_rgb_assignment_is_not_a_target_image_dump():
    code = 'def edit(image):\n    arr=np.asarray(image).copy()\n    arr[10,20]=(255,0,0)\n    return Image.fromarray(arr)\n'
    assert not looks_hardcoded(code)[0]


def test_external_execution_is_rejected():
    assert looks_hardcoded('import subprocess\ndef edit(image):\n    return image\n')[0]
