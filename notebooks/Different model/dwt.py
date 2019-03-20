# import numpy as np
# import matplotlib.pyplot as plt
#
# import pywt
# import pywt.data
#
#
# ecg = pywt.data.ecg()
#
# data1 = np.concatenate((np.arange(1, 400),
#                         np.arange(398, 600),
#                         np.arange(601, 1024)))
# x = np.linspace(0.082, 2.128, num=1024)[::-1]
# data2 = np.sin(40 * np.log(x)) * np.sign((np.log(x)))
#
# mode = pywt.Modes.smooth
#
#
# def plot_signal_decomp(data, w, title):
#     """Decompose and plot a signal S.
#     S = An + Dn + Dn-1 + ... + D1
#     """
#     w = pywt.Wavelet(w)
#     a = data
#     ca = []
#     cd = []
#     for i in range(5):
#         (a, d) = pywt.dwt(a, w, mode)
#         ca.append(a)
#         cd.append(d)
#
#     rec_a = []
#     rec_d = []
#
#     for i, coeff in enumerate(ca):
#         coeff_list = [coeff, None] + [None] * i
#         rec_a.append(pywt.waverec(coeff_list, w))
#
#     for i, coeff in enumerate(cd):
#         coeff_list = [None, coeff] + [None] * i
#         rec_d.append(pywt.waverec(coeff_list, w))
#
#     fig = plt.figure()
#     ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
#     ax_main.set_title(title)
#     ax_main.plot(data)
#     ax_main.set_xlim(0, len(data) - 1)
#
#     for i, y in enumerate(rec_a):
#         ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
#         ax.plot(y, 'r')
#         ax.set_xlim(0, len(y) - 1)
#         ax.set_ylabel("A%d" % (i + 1))
#
#     for i, y in enumerate(rec_d):
#         ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
#         ax.plot(y, 'g')
#         ax.set_xlim(0, len(y) - 1)
#         ax.set_ylabel("D%d" % (i + 1))
#
#
# plot_signal_decomp(data1, 'coif5', "DWT: Signal irregularity")
# plot_signal_decomp(data2, 'sym5',
#                    "DWT: Frequency and phase change - Symmlets5")
# plot_signal_decomp(ecg, 'sym5', "DWT: Ecg sample - Symmlets5")
#
#
# plt.show()
# Plot scaling and wavelet functions for db, sym, coif, bior and rbio families
#
# import itertools
#
# import matplotlib.pyplot as plt
#
# import pywt
#
#
# plot_data = [('db', (4, 3)),
#              ('sym', (4, 3)),
#              ('coif', (3, 2))]
#
#
# for family, (rows, cols) in plot_data:
#     fig = plt.figure()
#     fig.subplots_adjust(hspace=0.2, wspace=0.2, bottom=.02, left=.06,
#                         right=.97, top=.94)
#     colors = itertools.cycle('bgrcmyk')
#
#     wnames = pywt.wavelist(family)
#     i = iter(wnames)
#     for col in range(cols):
#         for row in range(rows):
#             try:
#                 wavelet = pywt.Wavelet(next(i))
#             except StopIteration:
#                 break
#             phi, psi, x = wavelet.wavefun(level=5)
#
#             color = next(colors)
#             ax = fig.add_subplot(rows, 2 * cols, 1 + 2 * (col + row * cols))
#             ax.set_title(wavelet.name + " phi")
#             ax.plot(x, phi, color)
#             ax.set_xlim(min(x), max(x))
#
#             ax = fig.add_subplot(rows, 2*cols, 1 + 2*(col + row*cols) + 1)
#             ax.set_title(wavelet.name + " psi")
#             ax.plot(x, psi, color)
#             ax.set_xlim(min(x), max(x))
#
# for family, (rows, cols) in [('bior', (4, 3)), ('rbio', (4, 3))]:
#     fig = plt.figure()
#     fig.subplots_adjust(hspace=0.5, wspace=0.2, bottom=.02, left=.06,
#                         right=.97, top=.94)
#
#     colors = itertools.cycle('bgrcmyk')
#     wnames = pywt.wavelist(family)
#     i = iter(wnames)
#     for col in range(cols):
#         for row in range(rows):
#             try:
#                 wavelet = pywt.Wavelet(next(i))
#             except StopIteration:
#                 break
#             phi, psi, phi_r, psi_r, x = wavelet.wavefun(level=5)
#             row *= 2
#
#             color = next(colors)
#             ax = fig.add_subplot(2*rows, 2*cols, 1 + 2*(col + row*cols))
#             ax.set_title(wavelet.name + " phi")
#             ax.plot(x, phi, color)
#             ax.set_xlim(min(x), max(x))
#
#             ax = fig.add_subplot(2*rows, 2*cols, 2*(1 + col + row*cols))
#             ax.set_title(wavelet.name + " psi")
#             ax.plot(x, psi, color)
#             ax.set_xlim(min(x), max(x))
#
#             row += 1
#             ax = fig.add_subplot(2*rows, 2*cols, 1 + 2*(col + row*cols))
#             ax.set_title(wavelet.name + " phi_r")
#             ax.plot(x, phi_r, color)
#             ax.set_xlim(min(x), max(x))
#
#             ax = fig.add_subplot(2*rows, 2*cols, 1 + 2*(col + row*cols) + 1)
#             ax.set_title(wavelet.name + " psi_r")
#             ax.plot(x, psi_r, color)
#             ax.set_xlim(min(x), max(x))
#
# plt.show()


import pywt
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
N = 256
x = np.arange(N)
y = np.sin(2*np.pi*x/64)
y = np.hstack((y,np.sin(2*np.pi*x/32)))
y = np.hstack((y,np.sin(2*np.pi*x/16)))
# for wav in pywt.wavelist():
#     print(wav)
#     try:
#         coef, freqs=pywt.cwt(y,np.arange(1,50),wavelet=wav)
#         ax1 = plt.subplot(211)
#         ax1.set_xlim([0, 2048])
#         ax1.plot(y)
#         ax2 = plt.subplot(212)
#         ax2.imshow(coef, aspect='auto', )
#         ax2.set_xlim([0, 2048])
#         plt.show()
#     except:
#         print()
# for wav in pywt.wavelist():
#     print(wav)
#     try:
#         coef, freqs=pywt.cwt(y,np.arange(1,50),wavelet=wav)
#         ax1 = plt.subplot(211)
#         ax1.set_xlim([0, 2048])
#         ax1.plot(y)
#         ax2 = plt.subplot(212)
#         ax2.imshow(coef, aspect='auto', )
#         ax2.set_xlim([0, 2048])
#         plt.show()
#     except:
#         print()
coef, freqs=pywt.cwt(y,np.arange(1,100),'morl')
fig = plt.figure(dpi=200)
ax1 = plt.subplot(411)
ax1.set_title("Исходный сигнал")
ax1.set_xlim([0,3*N])
ax1.plot(y)


ax2 = plt.subplot(412)
ax2.set_title('W(a,b)')
ax2.imshow(coef, aspect='auto',cmap='CMRmap')
ax2.set_xlim([0, 3*N])
plt.tight_layout()
# ax2.imshow(coef, extent=[-1, 1, 1, 31], cmap='CMRmap', aspect='auto',
#             vmax=abs(coef).max(), vmin=-abs(coef).max())


plt.savefig('wavlet.jpg')
plt.show()



rate = 10
tb = 10 / rate
sample_rate = 0.001
t = np.arange(0.0, tb, sample_rate)

fig = plt.figure(dpi=200)
y = 0.8*np.sin(40*2*np.pi*t) + 0.8*np.sin(10*2*np.pi*t)
coef, freqs=pywt.cwt(y,np.arange(1,200),'morl')
ax1 = plt.subplot(311)
ax1.set_title("Исходный сигнал")
ax1.set_xlim([0,3*N])
ax1.plot(y)


ax2 = plt.subplot(312)
ax2.set_title('W(a,b)')
ax2.imshow(coef, aspect='auto',cmap='CMRmap')
ax2.set_xlim([0, 3*N])
plt.tight_layout()
plt.show()

# t = np.linspace(-1, 1, 200, endpoint=False)
# sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
# widths = np.arange(1, 31)
# cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
# vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
# plt.show()