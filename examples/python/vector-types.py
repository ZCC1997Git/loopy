import numpy as np

import pyopencl as cl
import pyopencl.array

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

n = 15 * 10**6
a = cl.array.arange(queue, n, dtype=np.float32)

knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]")

knl = lp.set_options(knl, write_code=True)
knl = lp.split_iname(knl, "i", 4, slabs=(0, 1), inner_tag="vec")
knl = lp.split_array_axis(knl, "a,out", axis_nr=0, count=4)
knl = lp.tag_array_axes(knl, "a,out", "C,vec")

# 创建输出数组
out = cl.array.empty_like(a)

# 执行 kernel
knl(queue, a=a.reshape(-1, 4), out=out.reshape(-1, 4), n=n)

# 检查结果
expected = 2 * a.get()  # 预期结果：a 的每个元素乘以 2
actual = out.get()  # 实际结果

# 打印前10个结果对比
print("前10个结果对比:")
for i in range(10):
    print(f"  [{i:3d}] expected={expected[i]:.1f}, actual={actual[i]:.1f}, match={expected[i]==actual[i]}")

# 验证所有结果是否正确
if np.allclose(actual, expected):
    print("\n✓ 测试通过！所有结果正确")
else:
    print("\n✗ 测试失败！存在错误结果")
    # 找出第一个错误
    diff_indices = np.where(~np.isclose(actual, expected))[0]
    print(f"  第一个错误出现在索引: {diff_indices[0]}")
    print(f"  expected={expected[diff_indices[0]]}, actual={actual[diff_indices[0]]}")
