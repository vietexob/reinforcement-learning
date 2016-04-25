import pyximport, numpy
pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)

import super_fast_bot as sfb 
sfb.run_bbox()