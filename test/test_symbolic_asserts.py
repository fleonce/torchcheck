import unittest

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torchcheck import assert_shape, config

class SymbolicAssertTestCase(unittest.TestCase):

    def test_symbolic_assert(self):
        config.use_python_equivalents = True
        fake_env = FakeTensorMode(shape_env=ShapeEnv())
        t1 = torch.empty((1, 3))
        t1 = torch.empty((1, 3))
        with enable_python_dispatcher(), fake_env:
            t2 = fake_env.from_tensor(t1)
            self.assertRaises(
                RuntimeError,
                lambda: assert_shape(t2, (1, 3)),
                msg=f"No RuntimeError thrown by assert_shape(t2, (1, 3)), shape was {t2.shape}"
            )
            self.assertRaises(RuntimeError, lambda: assert_shape(t1, (1, t2.size(1))))
            self.assertEqual(assert_shape(t1, (1, 3)), 0)
            self.assertEqual(assert_shape(t2, t2.size()), 0)