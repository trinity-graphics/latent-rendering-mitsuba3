import pytest
import drjit as dr
import mitsuba as mi


def test01_construct(variant_scalar_rgb):
    # With default reconstruction filter
    film = mi.load_dict({'type': 'latfilm'})
    assert film is not None
    assert film.rfilter() is not None

    # With a provided reconstruction filter
    film = mi.load_dict({
        'type': 'latfilm',
        'filter': {
            'type': 'gaussian',
            'stddev': 18.5
        }
    })
    assert film is not None
    assert film.rfilter().radius() == (4 * 18.5)

    # Certain parameter values are not allowed
    with pytest.raises(RuntimeError):
        mi.load_dict({
            'type': 'latfilm',
            'component_format': "uint8",
        })
    with pytest.raises(RuntimeError):
        mi.load_dict({
            'type': 'latfilm',
            'pixel_format': "brga",
        })


def test02_crops(variant_scalar_rgb):
    film = mi.load_dict({
        'type': 'latfilm',
        'width': 32,
        'height': 21,
        'crop_width': 11,
        'crop_height': 5,
        'crop_offset_x': 2,
        'crop_offset_y': 3,
        'sample_border': True,
    })
    assert film is not None
    assert dr.all(film.size() == [32, 21])
    assert dr.all(film.crop_size() == [11, 5])
    assert dr.all(film.crop_offset() == [2, 3])
    assert film.sample_border()

    # Crop size doesn't adjust its size, so an error should be raised if the
    # resulting crop window goes out of bounds.
    incomplete = """<film version="3.0.0" type="latfilm">
            <integer name="width" value="32"/>
            <integer name="height" value="21"/>
            <integer name="crop_offset_x" value="30"/>
            <integer name="crop_offset_y" value="20"/>"""
    with pytest.raises(RuntimeError):
        film = mi.load_string(incomplete + "</film>")
    film = mi.load_string(incomplete + """
            <integer name="crop_width" value="2"/>
            <integer name="crop_height" value="1"/>
        </film>""")
    assert film is not None
    assert dr.all(film.size() == [32, 21])
    assert dr.all(film.crop_size() == [2, 1])
    assert dr.all(film.crop_offset() == [30, 20])


@pytest.mark.parametrize('file_format', ['exr'])
def test03_bitmap(variant_scalar_rgb, file_format, tmpdir):
    """
    Create a test image. Develop it to a few file format, each time reading
    it back and checking that contents are unchanged.
    """
    import numpy as np

    rng = np.random.default_rng(seed=12345 + ord(file_format[0]))
    # Note: depending on the file format, the alpha channel may be automatically removed.
    film = mi.load_dict({
        'type': 'latfilm',
        'width': 41,
        'height': 37,
        'filter': {'type': 'box'}
    })
    # Regardless of the output file format, values are stored as RGBAW (5 channels).
    contents = rng.uniform(size=(film.size()[1], film.size()[0], 17))
    # RGBE and will only reconstruct well images that have similar scales on
    # all channel (because exponent is shared between channels).

    # Use unit weights.
    contents[:, :, 16] = 1.0

    block = mi.ImageBlock(film.size(), [0, 0], 17, film.rfilter())
    for x in range(film.size()[1]):
        for y in range(film.size()[0]):
            block.put([y+0.5, x+0.5], contents[x, y, :])

    film.prepare([])
    film.put_block(block)

    filename = str(tmpdir.join('test_image.' + file_format))
    film.write(filename)

    # Read back and check contents
    other = mi.Bitmap(filename).convert(mi.Bitmap.PixelFormat.RGBAW, mi.Struct.Type.Float32, srgb_gamma=False)
    img   = mi.TensorXf(other)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(contents[:, :, :3])
        plt.subplot(1, 3, 2)
        plt.imshow(img[:, :, :3])
        plt.subplot(1, 3, 3)
        plt.imshow(dr.sum(dr.abs(img[:, :, :3] - contents[:, :, :3]), axis=2), cmap='coolwarm')
        plt.colorbar()
        plt.show()

    assert dr.allclose(img, contents, atol=1e-5)


@pytest.mark.parametrize('has_aovs', [False, True])
def test04_develop_and_bitmap(variants_all_rgb, has_aovs):
    aovs_channels = ['aov.r', 'aov.g', 'aov.b'] if has_aovs else []
    output_channels = ['c00','c01','c02','c03','c04','c05','c06','c07','c08','c09','c10','c11','c12','c13','c14','c15'] + aovs_channels

    film = mi.load_dict({
        'type': 'latfilm',
        'width': 3,
        'height': 5,
        # 'rfilter': { 'type': 'box' }
    })

    has_alpha = False

    res = film.size()
    block = mi.ImageBlock(res, [0, 0], (5 if has_alpha else 4) + len(aovs_channels), film.rfilter())

    if dr.is_jit_v(mi.Float):
        pixel_idx = dr.arange(mi.UInt32, dr.prod(res))
        x = pixel_idx % res[0]
        y = pixel_idx // res[0]

        pos = mi.Point2f(x, y) + 0.5
        v = [x, 2 * y, 0.1, 0.5]
        if has_alpha:
            v += [1.0]
        if has_aovs:
            v += [10 + x, 20 + y, 10.1]
        block.put(pos, v)
    else:
        for x in range(res[1]):
            for y in range(res[0]):
                v = [x, 2 * y, 0.1, 0.5]
                if has_alpha:
                    v += [1.0]
                if has_aovs:
                    v += [10 + x, 20 + y, 10.1]
                block.put([y + 0.5, x + 0.5], v)

    film.prepare(aovs_channels)
    film.put_block(block)

    image = film.develop()

    assert dr.prod(image.shape) == dr.prod(res) * (len(output_channels))

    data_bitmap = mi.Bitmap(image, mi.Bitmap.PixelFormat.MultiChannel, output_channels)
    bitmap = film.bitmap()
    assert dr.allclose(mi.TensorXf(bitmap), mi.TensorXf(data_bitmap))


def test05_without_prepare(variant_scalar_rgb):
    film = mi.load_dict({
        'type': 'latfilm',
        'width': 3,
        'height': 2,
    })

    with pytest.raises(RuntimeError, match=r'prepare\(\)'):
        _ = film.develop()


@pytest.mark.parametrize('develop', [False, True])
def test06_empty_film(variants_all_rgb, develop):
    film = mi.load_dict({
        'type': 'latfilm',
        'width': 3,
        'height': 2,
    })
    film.prepare([])

    if develop:
        image = dr.ravel(film.develop())
        assert dr.all((image == 0) | dr.isnan(image))
    else:
        image = mi.TensorXf(film.bitmap())
        assert dr.all((image == 0) | dr.isnan(image), axis=None)


def test07_luminance_alpha_mono(variants_all):
    film = mi.load_dict({
        'type': 'latfilm',
        'width': 3,
        'height': 2,
    })
    film.prepare([])

    image = mi.TensorXf(film.bitmap())

    assert image.shape[2] == 2
