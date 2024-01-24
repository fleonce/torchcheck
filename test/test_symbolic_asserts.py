import unittest
import logging
import torch
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torchcheck import assert_shape, config, C


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.DEBUG)

class SymbolicAssertTestCase(unittest.TestCase):

    def test_symbolic_assert(self):
        config.use_python_equivalents = True
        fake_env = FakeTensorMode(shape_env=ShapeEnv())
        t1 = torch.empty((1, 10))

        with enable_python_dispatcher(), fake_env:
            t2 = fake_env.from_tensor(t1)
            with self.assertRaises(
                    (AssertionError, RuntimeError),
                    msg=f"No RuntimeError thrown by assert_shape(t2, (1, 3)), "
                        f"shape was {t2.shape}"
            ):
                assert_shape(t2, (1, 10))
                assert_shape(t1, (1, t2.size(1)))
            self.assertEqual(assert_shape(t1, (1, 10)), 0)
            self.assertEqual(assert_shape(t2, t2.size()), 0)
