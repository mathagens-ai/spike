import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        f.close()
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except Exception:
            return False
        finally:
            try:
                os.remove(f.name)
            except OSError:
                pass
    return True

def cpp_flag(compiler):
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag): return flag
    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc', '/O2', '/arch:AVX2', '/openmp'],
        'unix': ['-O3', '-mavx2', '-fopenmp'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
            
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'snn_core',
        ['src/bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            '.'  # Allows preprocessor to find headers under snn-core/, snn-neural/, and snn-runtime/
        ],
        language='c++'
    ),
    Extension(
        'snn_training_cpp',
        ['snn-training/training_bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            '.',
            'snn-training'
        ],
        language='c++'
    ),
]

setup(
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.4'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
