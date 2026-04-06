## Test Output
2026-04-06 07:47:26.616196: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2026-04-06 07:47:26.658787: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-04-06 07:47:28.145102: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
E
======================================================================
ERROR: test_sample1 (test.TestModel)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/runner/work/example_versioning/example_versioning/test.py", line 22, in setUp
    self.restored_model = load_model(self.top_model_complete)
  File "/opt/hostedtoolcache/Python/3.9.25/x64/lib/python3.9/site-packages/keras/src/saving/saving_api.py", line 196, in load_model
    return legacy_h5_format.load_model_from_hdf5(
  File "/opt/hostedtoolcache/Python/3.9.25/x64/lib/python3.9/site-packages/keras/src/legacy/saving/legacy_h5_format.py", line 116, in load_model_from_hdf5
    f = h5py.File(filepath, mode="r")
  File "/opt/hostedtoolcache/Python/3.9.25/x64/lib/python3.9/site-packages/h5py/_hl/files.py", line 564, in __init__
    fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
  File "/opt/hostedtoolcache/Python/3.9.25/x64/lib/python3.9/site-packages/h5py/_hl/files.py", line 238, in make_fid
    fid = h5f.open(name, flags, fapl=fapl)
  File "h5py/_objects.pyx", line 56, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 57, in h5py._objects.with_phil.wrapper
  File "h5py/h5f.pyx", line 102, in h5py.h5f.open
FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'top_model_complete.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (errors=1)
