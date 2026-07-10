import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2_ops", "_version.py")).read())

# CUDA 12+ 已移除对 Kepler (sm_37) 的支持, 原来的 "3.7+PTX" 会触发
# nvcc fatal : Unsupported gpu architecture 'compute_37'.
# 这里如果用户未显式设置 TORCH_CUDA_ARCH_LIST, 则给一个不含 3.7 的默认值。
# 可按需自行 export TORCH_CUDA_ARCH_LIST="8.6" (或你的实际 GPU) 来缩短编译时间。
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    # 覆盖面较大的保守集合 (不含已弃用的 3.7)。如需更快编译请自行精简。
    os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
