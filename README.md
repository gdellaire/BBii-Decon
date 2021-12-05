# BBii-Decon
Projected Barzilai-Borwein Image Deconvolution with Infeasible Iterates (BBii-Decon)


The projected Barzilai-Borwein method of image deconvolution utilizing infeasible iterates (BBii-Decon), utilizes Barzilai-Borwein (BB) or projected BB (PBB) method and enforces a nonnegativity constraint, but allows for infeasible iterates between projections. This algorithm (BBii) results in faster convergence than the basic PBB method, while achieving better quality images, with reduced background than the unconstrained BB method (1). 

The code represented is based on the original BBii algorithm written in MatLab by Kathleen Fraser and Dirk Arnold, which was ported to python 3.8 by Graham Dellaire, Dirk Arnold and Kathleen Fraser for non-commercial use.



References: 
1) Kathleen Fraser, Dirk V. Arnold, and Graham Dellaire (2014). Projected Barzilai-Borwein
method with infeasible iterates for nonnegative least-squares image deblurring. In Proceedings
of the Eleventh Conference on Computer and Robot Vision (CRV 2014), Montreal, Canada, pp.
189--194. https://ieeexplore.ieee.org/abstract/document/6816842
