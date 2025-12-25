"""Tests for conv_nd."""

import pytest
import torch
import torch.nn.functional as F

from torch_conv_nd import conv_nd
from torch_conv_nd.conv import CONV_REGISTRY, CONV_TRANSPOSE_REGISTRY


@pytest.fixture
def disable_conv3d():
    orig, orig_t = CONV_REGISTRY.copy(), CONV_TRANSPOSE_REGISTRY.copy()
    del CONV_REGISTRY[3], CONV_TRANSPOSE_REGISTRY[3]
    yield
    CONV_REGISTRY.clear()
    CONV_REGISTRY.update(orig)
    CONV_TRANSPOSE_REGISTRY.clear()
    CONV_TRANSPOSE_REGISTRY.update(orig_t)


@pytest.fixture
def disable_conv2d_3d():
    orig, orig_t = CONV_REGISTRY.copy(), CONV_TRANSPOSE_REGISTRY.copy()
    CONV_REGISTRY.pop(2, None)
    CONV_REGISTRY.pop(3, None)
    CONV_TRANSPOSE_REGISTRY.pop(2, None)
    CONV_TRANSPOSE_REGISTRY.pop(3, None)
    yield
    CONV_REGISTRY.clear()
    CONV_REGISTRY.update(orig)
    CONV_TRANSPOSE_REGISTRY.clear()
    CONV_TRANSPOSE_REGISTRY.update(orig_t)


class TestForwardNative:
    def test_conv1d(self):
        x, w = torch.randn(2, 4, 32), torch.randn(8, 4, 3)
        expected = F.conv1d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-1,), stride=2, padding=1)
        torch.testing.assert_close(result, expected)

    def test_conv2d(self):
        x, w = torch.randn(2, 4, 16, 16), torch.randn(8, 4, 3, 3)
        expected = F.conv2d(x, w, stride=2, padding=1, dilation=2)
        result = conv_nd(x, w, dim=(-2, -1), stride=2, padding=1, dilation=2)
        torch.testing.assert_close(result, expected)

    def test_conv3d(self):
        x, w = torch.randn(2, 4, 8, 8, 8), torch.randn(8, 4, 3, 3, 3)
        expected = F.conv3d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=1)
        torch.testing.assert_close(result, expected)


class TestTransposeNative:
    def test_conv_transpose1d(self):
        x, w = torch.randn(2, 4, 8), torch.randn(4, 8, 3)
        expected = F.conv_transpose1d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-1,), stride=2, padding=1, transposed=True)
        torch.testing.assert_close(result, expected)

    def test_conv_transpose2d(self):
        x, w = torch.randn(2, 4, 8, 8), torch.randn(4, 8, 3, 3)
        expected = F.conv_transpose2d(x, w, stride=2, padding=1, output_padding=1)
        result = conv_nd(x, w, dim=(-2, -1), stride=2, padding=1, output_padding=1, transposed=True)
        torch.testing.assert_close(result, expected)

    def test_conv_transpose3d(self):
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(4, 8, 3, 3, 3)
        expected = F.conv_transpose3d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=1, transposed=True)
        torch.testing.assert_close(result, expected)


class TestRecursiveForward:
    def test_conv3d_via_conv2d(self, disable_conv3d):  # noqa: ARG002
        x, w = torch.randn(2, 4, 8, 8, 8), torch.randn(8, 4, 3, 3, 3)
        expected = F.conv3d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_conv3d_via_conv1d(self, disable_conv2d_3d):  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(8, 4, 3, 3, 3)
        expected = F.conv3d(x, w, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), padding=1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_asymmetric_params(self, disable_conv3d):  # noqa: ARG002
        x, w = torch.randn(2, 4, 12, 14, 16), torch.randn(8, 4, 3, 5, 3)
        stride, padding, dilation = (2, 1, 2), (1, 2, 1), (1, 2, 1)
        expected = F.conv3d(x, w, stride=stride, padding=padding, dilation=dilation)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=stride, padding=padding, dilation=dilation)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestRecursiveTranspose:
    def test_transpose3d_via_conv2d(self, disable_conv3d):  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(4, 8, 3, 3, 3)
        expected = F.conv_transpose3d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=1, transposed=True)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_transpose3d_with_output_padding(self, disable_conv3d):  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(4, 8, 3, 3, 3)
        expected = F.conv_transpose3d(x, w, stride=2, padding=1, output_padding=1)
        result = conv_nd(
            x, w, dim=(-3, -2, -1), stride=2, padding=1, output_padding=1, transposed=True
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_transpose3d_output_padding_exceeds_padding(self, disable_conv3d):  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(4, 8, 3, 3, 3)
        expected = F.conv_transpose3d(x, w, stride=2, padding=0, output_padding=1)
        result = conv_nd(
            x, w, dim=(-3, -2, -1), stride=2, padding=0, output_padding=1, transposed=True
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_transpose3d_asymmetric(self, disable_conv3d):  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 8, 10), torch.randn(4, 8, 3, 5, 3)
        stride, padding = (2, 1, 2), (1, 2, 1)
        expected = F.conv_transpose3d(x, w, stride=stride, padding=padding)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=stride, padding=padding, transposed=True)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestComplex:
    def test_both_complex(self):
        x = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
        w = torch.randn(8, 4, 3, 3, dtype=torch.complex64)
        result = conv_nd(x, w, dim=(-2, -1), padding=1)
        expected = torch.complex(
            F.conv2d(x.real, w.real, padding=1) - F.conv2d(x.imag, w.imag, padding=1),
            F.conv2d(x.real, w.imag, padding=1) + F.conv2d(x.imag, w.real, padding=1),
        )
        torch.testing.assert_close(result, expected)

    def test_complex_x_only(self):
        x = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
        w = torch.randn(8, 4, 3, 3)
        result = conv_nd(x, w, dim=(-2, -1), padding=1)
        expected = torch.complex(F.conv2d(x.real, w, padding=1), F.conv2d(x.imag, w, padding=1))
        torch.testing.assert_close(result, expected)

    def test_complex_w_only(self):
        x = torch.randn(2, 4, 16, 16)
        w = torch.randn(8, 4, 3, 3, dtype=torch.complex64)
        result = conv_nd(x, w, dim=(-2, -1), padding=1)
        expected = torch.complex(F.conv2d(x, w.real, padding=1), F.conv2d(x, w.imag, padding=1))
        torch.testing.assert_close(result, expected)

    def test_complex_transpose(self):
        x = torch.randn(2, 4, 8, 8, dtype=torch.complex64)
        w = torch.randn(4, 8, 3, 3, dtype=torch.complex64)
        result = conv_nd(x, w, dim=(-2, -1), padding=1, transposed=True)
        expected = torch.complex(
            F.conv_transpose2d(x.real, w.real, padding=1)
            - F.conv_transpose2d(x.imag, w.imag, padding=1),
            F.conv_transpose2d(x.real, w.imag, padding=1)
            + F.conv_transpose2d(x.imag, w.real, padding=1),
        )
        torch.testing.assert_close(result, expected)


class TestGroups:
    def test_groups_forward(self):
        x, w = torch.randn(2, 8, 16, 16), torch.randn(16, 4, 3, 3)
        expected = F.conv2d(x, w, groups=2, padding=1)
        result = conv_nd(x, w, dim=(-2, -1), groups=2, padding=1)
        torch.testing.assert_close(result, expected)

    def test_groups_transpose(self):
        x, w = torch.randn(2, 8, 8, 8), torch.randn(8, 4, 3, 3)
        expected = F.conv_transpose2d(x, w, groups=2, padding=1)
        result = conv_nd(x, w, dim=(-2, -1), groups=2, padding=1, transposed=True)
        torch.testing.assert_close(result, expected)

    def test_groups_recursive(self, disable_conv3d):  # noqa: ARG002
        x, w = torch.randn(2, 8, 6, 6, 6), torch.randn(16, 4, 3, 3, 3)
        expected = F.conv3d(x, w, groups=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), groups=2, padding=1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestDimLayouts:
    def test_channel_last(self):
        x, w = torch.randn(2, 16, 16, 4), torch.randn(8, 4, 3, 3)
        expected = F.conv2d(x.permute(0, 3, 1, 2), w, padding=1).permute(0, 2, 3, 1)
        result = conv_nd(x, w, dim=(1, 2), channel_dim=-1, padding=1)
        torch.testing.assert_close(result, expected)

    def test_extra_batch_dims(self):
        x, w = torch.randn(2, 3, 4, 16, 16), torch.randn(8, 4, 3, 3)
        expected = F.conv2d(x.reshape(6, 4, 16, 16), w, padding=1).reshape(2, 3, 8, 16, 16)
        result = conv_nd(x, w, dim=(-2, -1), channel_dim=2, padding=1)
        torch.testing.assert_close(result, expected)

    def test_non_contiguous_spatial(self):
        x, w = torch.randn(2, 8, 4, 8), torch.randn(6, 4, 3, 3)
        expected = F.conv2d(x.permute(0, 2, 1, 3), w, padding=1).permute(0, 2, 1, 3)
        result = conv_nd(x, w, dim=(1, 3), channel_dim=2, padding=1)
        torch.testing.assert_close(result, expected)


class TestAutograd:
    def test_backward_forward(self):
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        w = torch.randn(8, 4, 3, 3, requires_grad=True)
        conv_nd(x, w, dim=(-2, -1), padding=1).sum().backward()
        assert x.grad is not None and w.grad is not None

    def test_backward_transpose(self):
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        w = torch.randn(4, 8, 3, 3, requires_grad=True)
        conv_nd(x, w, dim=(-2, -1), padding=1, transposed=True).sum().backward()
        assert x.grad is not None and w.grad is not None

    def test_gradcheck_recursive(self, disable_conv3d):  # noqa: ARG002
        x = torch.randn(1, 2, 5, 5, 5, dtype=torch.float64, requires_grad=True)
        w = torch.randn(3, 2, 3, 3, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda a, b: conv_nd(a, b, dim=(-3, -2, -1), padding=1), (x, w)
        )

    def test_gradcheck_transpose_recursive(self, disable_conv3d):  # noqa: ARG002
        x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(2, 3, 3, 3, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda a, b: conv_nd(a, b, dim=(-3, -2, -1), padding=1, transposed=True), (x, w)
        )

    def test_gradcheck_complex(self):
        x = torch.randn(1, 2, 5, 5, dtype=torch.complex128, requires_grad=True)
        w = torch.randn(3, 2, 3, 3, dtype=torch.complex128, requires_grad=True)
        assert torch.autograd.gradcheck(lambda a, b: conv_nd(a, b, dim=(-2, -1), padding=1), (x, w))


class TestEdgeCases:
    def test_kernel_1x1(self):
        x, w = torch.randn(2, 4, 8, 8), torch.randn(8, 4, 1, 1)
        torch.testing.assert_close(conv_nd(x, w, dim=(-2, -1)), F.conv2d(x, w))

    def test_stride_larger_than_kernel(self):
        x, w = torch.randn(2, 4, 16, 16), torch.randn(8, 4, 3, 3)
        expected = F.conv2d(x, w, stride=5, padding=1)
        result = conv_nd(x, w, dim=(-2, -1), stride=5, padding=1)
        torch.testing.assert_close(result, expected)

    def test_no_batch_dim(self):
        x, w = torch.randn(4, 8, 8), torch.randn(8, 4, 3, 3)
        expected = F.conv2d(x.unsqueeze(0), w, padding=1).squeeze(0)
        result = conv_nd(x, w, dim=(-2, -1), channel_dim=0, padding=1)
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
class TestCompile:
    def test_compile_forward(self):
        fn = torch.compile(conv_nd)
        x, w = torch.randn(2, 4, 16, 16), torch.randn(8, 4, 3, 3)
        torch.testing.assert_close(
            fn(x, w, dim=(-2, -1), padding=1), conv_nd(x, w, dim=(-2, -1), padding=1)
        )

    def test_compile_transpose(self):
        fn = torch.compile(conv_nd)
        x, w = torch.randn(2, 4, 8, 8), torch.randn(4, 8, 3, 3)
        torch.testing.assert_close(
            fn(x, w, dim=(-2, -1), padding=1, transposed=True),
            conv_nd(x, w, dim=(-2, -1), padding=1, transposed=True),
        )
