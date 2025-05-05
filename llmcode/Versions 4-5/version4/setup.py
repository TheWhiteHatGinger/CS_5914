from setuptools import setup, Extension
import pybind11
import os

# Path to Eigen (adjust as needed)
eigen_include = "/usr/include/eigen3"  # Example path for Ubuntu

ext_modules = [
    Extension(
        "amg_solver",
        ["amg_solver.cpp"],
        include_dirs=[pybind11.get_include(), eigen_include],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]

setup(
    name="amg_solver",
    ext_modules=ext_modules,
)
