import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2)

def plot_hist(sub, title, txt, img):
    text, image = np.load(txt), np.load(img)
    sub.hist(image, bins=50, alpha=0.5, label='Image', density=True)
    sub.hist(text, bins=50, alpha=0.5, label='Text', density=True)    
    sub.set_title(title)
    sub.legend()

plot_hist(axs[0], "λ = 0", "norm_med_eucl_sq_noln_text_l2_norms.npy", "norm_med_eucl_sq_noln_image_l2_norms.npy")
plot_hist(axs[1], "λ = 0.1", "norm_med_eucli-squ_noln_logit_neg1_entw01_text_l2_norms.npy", "norm_med_eucli-squ_noln_logit_neg1_entw01_image_l2_norms.npy")

axs.flat[0].set(ylabel='Percentage of instances')
axs.flat[1].set(xlabel='L2 Norm', ylabel='Percentage of instances')

plt.tight_layout()
plt.savefig('norm_distribution_plot.png')
plt.close()
