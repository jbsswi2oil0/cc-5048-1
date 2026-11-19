import numpy as np
import pytest

from binaural.convolution import PartitionedConvolver, measure_pipeline_latency
from binaural.hrtf import ProceduralHRTF
from binaural.renderer import BinauralRenderer

BLOCK_SIZES = [64, 128, 256, 512]


def reference_convolution(x, h):
    nfft = 1 << int(np.ceil(np.log2(len(x) + len(h))))
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("hrir_length", [10, 64, 128, 256, 512, 700])
def test_matches_direct_convolution_with_one_block_delay(block_size, hrir_length):
    rng = np.random.default_rng(block_size + hrir_length)
    h = rng.standard_normal(hrir_length)
    x = rng.standard_normal(block_size * 8)
    conv = PartitionedConvolver(h, block_size)
    got = np.concatenate([
        conv.process_block(x[k:k + block_size])
        for k in range(0, len(x), block_size)])
    ref = reference_convolution(x, h)
    valid = got.shape[0] - block_size
    assert got.shape == x.shape
    np.testing.assert_allclose(got[block_size:block_size + valid],
                               ref[:valid], atol=1e-8)


def test_block_size_must_be_power_of_two():
    with pytest.raises(ValueError):
        PartitionedConvolver(np.ones(8), block_size=100)


def test_raises_on_wrong_block_length():
    conv = PartitionedConvolver(np.ones(8), block_size=128)
    with pytest.raises(ValueError):
        conv.process_block(np.zeros(10))


def test_impulse_latency_is_one_block():
    block_size = 256
    h = np.zeros(64)
    h[0] = 1.0
    conv = PartitionedConvolver(h, block_size)
    first = conv.process_block(
        np.r_[1.0, np.zeros(block_size - 1)])
    second = conv.process_block(np.zeros(block_size))
    assert np.max(np.abs(first)) == 0.0
    assert second[0] == pytest.approx(1.0)
    assert conv.algorithm_latency_samples == block_size


def test_reset_restores_identical_output():
    rng = np.random.default_rng(1)
    h, x = rng.standard_normal(200), rng.standard_normal(256 * 6)
    conv = PartitionedConvolver(h, 256)
    run = lambda: np.concatenate([
        conv.process_block(x[k:k + 256]) for k in range(0, len(x), 256)])
    first = run()
    conv.reset()
    second = run()
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_pipeline_latency_measurement():
    hrtf = ProceduralHRTF()
    renderer = BinauralRenderer(hrtf, block_size=256,
                                apply_distance_gain=False)
    renderer._current_query = None
    renderer.set_direction(30, 0, 1.2)
    report = measure_pipeline_latency(renderer, hrtf.sample_rate)
    assert report["matches_documentation"]
    assert report["block_latency_samples_left"] == 256
    assert report["total_arrival_samples_left"] >= 256
