import sunpy.io.fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

'''
Master flat
Inputs
    Single file of stack of flats 𝑁×𝑌×𝑊 
    Master dark file 𝑌×𝑊 
    Corrected fringe flat 𝑌×𝑊

Outputs
    Master flat file saved in Level-1 directory 𝑌×𝑊 
    Row and column shifts saved in Level-1 directory as text 
    Line profile saved in Level-1 directory as text 

Process
    Stack of 𝑁 number of flat files, master dark are loaded 
    Flats are averaged then master dark is subtracted 
    X inclination is corrected by shifting the columns 
    Column shift values are obtained by tracing the slit profile 
    Y inclination is corrected by shifting the rows 
    Row shift values are obtained by tracing a fine spectral line 
    For this trace, flat is corrected by fringe flat to reduce the contrast of fringes due to etaloning effect 
    After X, Y inclination corrections, image is aligned and medial spectral line is calculated 
    Half of rows from middle of top and bottom beams of the image with reduced fringe contrast are used for this 
    Each row in flat is divided with the line profile 
    Result is formatted and saved as FLATMASTER
'''

def generate_master_flat(base_path, file_format, num_of_files=18, dark_master_path=''):
    filename_list = list()

    for i in range(1, num_of_files+1):
        filename_list.append(
            file_format.format(i)
        )

    fits_list = list()

    for _filename in filename_list:
        fits_list.append(
            sunpy.io.fits.read(base_path+_filename)[0].data
        )

    averaged_flat_header = sunpy.io.fits.read(base_path+_filename)[0].header

    averaged_flat = np.average(fits_list, axis=0)

    dark_image = sunpy.io.fits.read(base_path+dark_master_path)[0][0]
    dark_corrected_averaged_flat = (averaged_flat - dark_image)[0]


    ########### X Tilt Correction ###################################

    max_shift, extent, up_sampling = 8, 10, 10

    h_pix = dark_corrected_averaged_flat.shape[1]

    display = dark_corrected_averaged_flat.copy()

    plt.imshow(display, cmap='gray')

    point = list(map(int, plt.ginput(1)[0]))

    plt.close()

    ref_col, y_beg, y_end = point[0], point[1] - extent / 2, point[1] + extent / 2

    slit_ref = display[int(y_beg):int(y_end), ref_col]

    normalised_slit = (slit_ref - slit_ref.mean()) / slit_ref.std()  # Normalize the selected slit profile

    weights = np.zeros(shape=h_pix)

    shift_ver = np.zeros(shape=h_pix)

    for j in np.arange(h_pix):
        _slit = display[int(y_beg) - max_shift:int(y_end) + max_shift, j]
        weights[j] = abs(_slit.mean())
        _slit_normalised = (_slit - _slit.mean()) / _slit.std()  # Normalize the selected slit profile
        # Correlate
        correlation = np.correlate(
            scipy.ndimage.zoom(
                _slit_normalised,
                up_sampling
            ),
            scipy.ndimage.zoom(
                normalised_slit,
                up_sampling
            ),
            mode='valid'
        )
        shift_ver[j] = np.argmax(correlation)  # Get the index with maximum correlation

    shift_ver = shift_ver / up_sampling - max_shift  # Convert the index to pixel shift

    xfit_xinc = np.argwhere(abs(shift_ver) < max_shift)

    yfit_yinc = shift_ver[xfit_xinc]

    line_fit = np.polyfit(xfit_xinc.ravel(), yfit_yinc.ravel(), 1, w=np.nan_to_num(weights)[xfit_xinc].ravel())  # Fit a line

    shift_ver_fit = line_fit[0] * np.arange(h_pix) + line_fit[1]

    shift_ver_apply = -shift_ver_fit

    plt.plot(xfit_xinc, yfit_yinc, 'k-', shift_ver_fit, 'k-')

    plt.show()

    x_corrected = dark_corrected_averaged_flat.copy()

    for i in np.arange(h_pix):  # Calculate the shift corrected array
        scipy.ndimage.shift(dark_corrected_averaged_flat[:, i], shift_ver_apply[i], x_corrected[:, i], mode='nearest')

    plt.imshow(x_corrected, cmap='gray')

    plt.show()

    ######################################################################################################################

    ######################## Y Tilt Correction ############################################################################
    max_shift, extent, up_sampling = 10, 20, 10

    v_pix = dark_corrected_averaged_flat.shape[0]

    display = x_corrected.copy()

    plt.imshow(display, cmap='gray')

    point = list(map(int, plt.ginput(1)[0]))

    plt.close()

    ref_row, x_beg, x_end = int(point[1]), int(point[0] - extent / 2), int(point[0] + extent / 2)

    line_ref = np.mean(display[int(ref_row) - 10:int(ref_row) + 10, int(x_beg):int(x_end)], 0)  # Reference line profile

    normalised_line = (line_ref - line_ref.mean())/line_ref.std()  # Normalize the reference line profile

    weights = np.zeros(shape=v_pix)

    shift_hor = np.zeros(shape=v_pix)

    for j in np.arange(v_pix):
        if 5 <= j < v_pix-5:
            _line = np.mean(display[j - 5:j + 5, x_beg - max_shift:x_end + max_shift], axis=0)
        else:
            _line = display[j, x_beg - max_shift:x_end + max_shift]
        weights[j] = abs(_line.mean())
        _line_normalised = (_line - _line.mean()) / _line.std()  # Normalize the selected line profile
        correlation = np.correlate(
            scipy.ndimage.zoom(
                _line_normalised,
                up_sampling
            ),
            scipy.ndimage.zoom(
                normalised_line,
                up_sampling
            ),
            mode='valid'
        )
        shift_hor[j] = np.argmax(correlation)  # Get the index with maximum correlation
    shift_hor = shift_hor / up_sampling - max_shift
    xfit_yinc = np.argwhere((abs(shift_hor) < max_shift))
    yfit_yinc = shift_hor[xfit_yinc]
    polynomial_fit = np.polyfit(xfit_yinc.ravel(), yfit_yinc.ravel(), 2, w=np.nan_to_num(weights)[xfit_yinc].ravel())  # Fit a line
    shift_hor_fit = polynomial_fit[0] * np.arange(v_pix) ** 2 + polynomial_fit[1] * np.arange(v_pix) + np.arange(v_pix)[2]  # Use the equation to calculate shifts
    shift_hor_apply = -shift_hor_fit
    plt.plot(xfit_yinc, yfit_yinc, 'k-', shift_hor_fit, 'k-')
    plt.show()

    y_corrected = x_corrected.copy()

    for i in np.arange(v_pix):
        scipy.ndimage.shift(
            x_corrected[i, :],
            shift_hor_apply[i],
            y_corrected[i, :],
            mode='nearest'
        )

    plt.imshow(y_corrected, cmap='gray')

    plt.show()

    #######################################################################################################################


    ################ Line Profile removal###############################################################################

    rows_1 = np.arange(100, 400)
    rows_2 = np.arange(600, 900)

    cropped_array = np.append(y_corrected[rows_1], y_corrected[rows_2], axis=0)

    line_median = np.median(cropped_array, axis=0)

    line_median_normalised = line_median / line_median.max()

    line_filter = scipy.ndimage.gaussian_filter1d(line_median_normalised, 2)

    flat_line_removed = np.divide(y_corrected, line_filter)

    plt.imshow(flat_line_removed, cmap='gray')

    plt.show()

    return flat_line_removed, averaged_flat_header


def do():
    base_path = '/Users/harshmathur/Documents/CourseworkRepo/Spectroscopy Data/'
    file_format = 'Flat_Prom_20170509_085123070_FORWARD_{}.fits'
    num_of_files = 18
    dark_master_path = 'level1/dark_master.fits'
    data, header = generate_master_flat(base_path, file_format, num_of_files, dark_master_path)
    sunpy.io.fits.write('flat_master.fits', data=data, header=header)


if __name__ == '__main__':
    do()
