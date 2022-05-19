import matplotlib.pyplot as plt

AP_cellpose = [0.81, 0.80, 0.75, 0.71, 0.65, 0.57, 0.46, 0.32, 0.14, 0.01, 0]
AP_unet = [0.067, 0.050, 0.034, 0.023, 0.011, 0.0056, 0.0025, 0.0011, 0.00078, 0, 0]
IoU_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
plt.plot(IoU_list, AP_cellpose, marker='o', mec='r', mfc='w', label='Cellpose')
plt.plot(IoU_list, AP_unet, marker='*', ms=10, label='U-Net')
plt.legend()  # 让图例生效

plt.margins(0)
plt.subplots_adjust(bottom=0.10)
plt.xlabel('IoU')  # X轴标签
plt.ylabel("Average Precision")  # Y轴标签
plt.show()