import imp
import os
if os.path.exists('libtest_ct_python.so'):
    imp.load_dynamic('test_ct_python','libtest_ct_python.so')
else:
    imp.load_dynamic('test_ct_python','libtest_ct_pythond.so')
