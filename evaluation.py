import numpy as np
import denoising_algorithm as DA
import scipy as sp

def evaluation(test_data, TV_A, TL_A, BH_A):
    N = 100
    evaluated_data = []
    MSE_u_table = []
    MGE_table = []
    for i in range(N):
        sample_i = test_data[i]
        image_m, image_n, _ = sample_i.shape
        m_blocks = image_m / 16
        n_blocks = image_n / 16

        evaluated_sample = np.zeros((image_m, image_n, 8))

        image_i = sample_i[..., 0]
        noisy_image_i = sample_i[..., 1]

        for r in range(m_blocks):
            for s in range(n_blocks):
                block = image_i[16*r : 16*r + 15, 16*s : 16*s + 15]
                block_shape = block.shape

                noisy_block = noisy_image_i[16 * r: 16 * r + 15, 16 * s: 16 * s + 15]

                g_bar = np.ones(16*16 + 1)
                vectorized_noisy_block = noisy_block.flatten(order='F')
                g_bar[0:16*16] = vectorized_noisy_block

                TV_alpha = g_bar.T @ TV_A @ g_bar
                TL_alpha = g_bar.T @ TL_A @ g_bar
                BH_alpha = g_bar.T @ BH_A @ g_bar

                TV_denoised_sample, TV_denoise_gap =\
                    DA.RCP('TV', 1/(np.sqrt(8)), 1/(np.sqrt(8)), 10000, 1e-7, 100, noisy_block, TV_alpha)
                TL_denoised_sample, TL_denoise_gap = \
                    DA.RCP('TL', 1 / (np.sqrt(8)), 1 / (np.sqrt(8)), 10000, 1e-7, 100, noisy_block, TL_alpha)
                BH_denoised_sample, BH_denoise_gap = \
                    DA.RCP('BH', 1 / (np.sqrt(8)), 1 / (np.sqrt(8)), 10000, 1e-7, 100, noisy_block, BH_alpha)

                TV_SE = np.sum((block - TV_denoised_sample) ** 2)
                TL_SE = np.sum((block - TL_denoised_sample) ** 2)
                BH_SE = np.sum((block - BH_denoised_sample) ** 2)

                MSE_u_sample = np.zeros(3)
                MGE_sample = np.zeros(3)

                MSE_u_sample[0] = TV_SE
                MSE_u_sample[1] = TL_SE
                MSE_u_sample[2] = BH_SE

                MGE_sample[0] = TV_denoise_gap
                MGE_sample[1] = TL_denoise_gap
                MGE_sample[2] = BH_denoise_gap

                MSE_u_table.append(MSE_u_sample)
                MGE_table.append(MGE_sample)

                evaluated_sample[16*r : 16*r + 15, 16*s : 16*s + 15, 0] = block
                evaluated_sample[16*r : 16*r + 15, 16*s : 16*s + 15, 1] = noisy_block
                evaluated_sample[16*r : 16*r + 15, 16*s : 16*s + 15, 2] = TV_alpha*np.ones(block_shape)
                evaluated_sample[16*r : 16*r + 15, 16*s : 16*s + 15, 3] = TV_denoised_sample
                evaluated_sample[16*r : 16*r + 15, 16*s : 16*s + 15, 4] = TL_alpha*np.ones(block_shape)
                evaluated_sample[16*r : 16*r + 15, 16*s : 16*s + 15, 5] = TL_denoised_sample
                evaluated_sample[16*r : 16*r + 15, 16*s : 16*s + 15, 6] = BH_alpha*np.ones(block_shape)
                evaluated_sample[16*r : 16*r + 15, 16*s : 16*s + 15, 7] = BH_denoised_sample

        evaluated_data.append(evaluated_sample)


    sp.io.savemat('data/output/evaluated data/evaluation_results.mat', {
        'evaluated_data': evaluated_data,
        'MSE_table': MSE_u_table,
        'MGE_table': MGE_table,
    })