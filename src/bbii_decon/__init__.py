
__version__ = "0.0.1"

import numpy
from napari_plugin_engine import napari_hook_implementation

# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    # we can return a single function
    # or a tuple of (function, magicgui_options)
    # or a list of multiple functions with or without options, as shown here:
    return [bbii_deconvolution]

# This is the function we call from napari. The type annotations and default values allow to build a user interface
# automatically.
try:
    import cupy
    def bbii_deconvolution(
            PSF: "napari.types.ImageData",
            blurry_image: "napari.types.ImageData",
            number_of_iterations: int = 10,
            tau: float = 1.0e-08,
            rho: float = 0.98,
            use_gpu: bool = False
    ) -> "napari.types.ImageData":
        return bbii(PSF, blurry_image, number_of_iterations, tau, rho, use_gpu)[0]
except ImportError:
    def bbii_deconvolution(
            PSF: "napari.types.ImageData",
            blurry_image: "napari.types.ImageData",
            number_of_iterations: int = 10,
            tau: float = 1.0e-08,
            rho: float = 0.98
    ) -> "napari.types.ImageData":
        return bbii(PSF, blurry_image, number_of_iterations, tau, rho)[0]


# BBii function; Source: Fraser, Arnold, Dellaire 2014 https://ieeexplore.ieee.org/abstract/document/6816842
def bbii(PSF, blurry_image, number_of_iterations, tau, rho, use_gpu=False):
    """The projected Barzilai-Borwein method of image deconvolution utilizing infeasible iterates (BBii-Decon),
    utilizes Barzilai-Borwein (BB) or projected BB (PBB) method and enforces a nonnegativity constraint, but allows for
    infeasible iterates between projections. This algorithm (BBii) results in faster convergence than the basic PBB
    method, while achieving better quality images, with reduced background than the unconstrained BB method [1].

    Parameters
    ----------
    PSF : ndarray
        point spread function
    blurry_image : ndarray
        blurry , noisy image
    number_of_iterations: int
        number of iterations to run the algorithm
    tau: float
        initial threshold for when to project (tau < 0) -- should this really be less than 0??
    rho: float
        scaling factor by which tau decreases over time (0 <= rho <= 1)
    use_gpu: bool
        Use GPU-acceleration using cupy

    Returns
    -------
    f - deconvolved image
    alpha - step sizes
    proj - record of projections (1 or 0 for projection or no projection)

    See also
    --------
    ..[1] https://ieeexplore.ieee.org/abstract/document/6816842
    """
    from pypher.pypher import psf2otf

    import numpy
    if use_gpu:
        import cupy as cp
        np = cp
    else:
        np = numpy


    # convert to numpy arrays
    PSF = numpy.array(PSF)
    b = numpy.array(blurry_image)
    K = psf2otf(PSF, b.shape)

    PSF = np.asarray(PSF)
    b = np.asarray(b)
    K = np.asarray(K)

    iter = number_of_iterations

    numel_b = 1  # number of elements in b (use this later)
    for dim in b.shape:
        numel_b *= dim

    f = np.zeros(b.shape)
    alpha = np.zeros([iter + 1])
    proj = np.zeros([iter])

    # not in original -- initialize r and med
    r = np.zeros([iter])
    med = np.zeros([iter])


    F = np.fft.fftn(f)
    B = np.fft.fftn(b)

    G = np.multiply(np.conj(K), (np.multiply(K, F) - B))

    g = np.real(np.fft.ifftn(G))
    Kg = np.real(np.fft.ifftn(np.multiply(K, G)))

    alpha[0] = np.matmul(g.flatten().T, g.flatten()) / np.matmul(Kg.flatten().T, Kg.flatten())

    for k in range(1, iter):

        if k > 10 and proj[k - 1] != 0:  # projection on last step
            alpha[k] = np.matmul(g.flatten().T, g.flatten()) / np.matmul(Kg.flatten().T,
                                                                         Kg.flatten())  # use steepest descent step size

        f = f - alpha[k] * g  # update solution

        # how many infeasible pixels?
        less = f < 0
        if np.sum(less) > 0:
            neg = np.multiply(f, less)
            # r is average squared value of all negative pixel values, divided by average squared value of all the pixels
            r[k] = (np.sum(np.multiply(neg, neg)) / np.sum(less)) / (np.sum(np.multiply(f, f)) / numel_b)
        else:
            r[k] = 0

        # step size for next iteration
        alpha[k + 1] = np.matmul(g.flatten().T, g.flatten()) / np.matmul(Kg.flatten().T, Kg.flatten())

        # finding the median r value
        if k >= 10:  # for the first 10 steps
            med[k] = np.median(r[k - 10:k])  # median over last 10 steps

        if k > 10 and med[k] > tau:  # project if median(r) > tau
            f[f < 0] = 0
            num = 3  # ??
            proj[k] = 1
            tau = rho * tau  # rho is factor by which tau decreases over time
        else:
            num = 2
            proj[k] = 0

        if num == 3:  # if projection , use gradient definition
            F = np.fft.fftn(f)  # requires an addition FFT
            G = np.multiply(np.conj(K), (np.multiply(K, F) - B))

        else:  # otherwise use iterative update
            G = G - alpha[k] * (np.multiply(np.multiply(np.conj(K), K), G))

        g = np.real(np.fft.ifftn(G))
        Kg = np.real(np.fft.ifftn(np.multiply(K, G)))

        if k == iter - 1:  # always project on last iteration
            f[f < 0] = 0  # so output solution is feasible

    if numpy == np:
        return [f, alpha, proj]
    else:
        return [
            numpy.asarray(f.get()),
            numpy.asarray(alpha.get()),
            numpy.asarray(proj.get())]

