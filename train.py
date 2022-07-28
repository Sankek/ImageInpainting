import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output

from utils import smooth1d, tensor2image, save_state


def train_step(optimizer, loss_terms, losses_storage, loss_terms_storage):
    if optimizer:
        optimizer.zero_grad()
    loss = sum(loss_terms)
    loss_terms = [term.item() for term in loss_terms]
    loss_terms_storage.append(loss_terms)
    losses_storage.append(loss.item())
    loss.backward()
    if optimizer:
        optimizer.step()

    
def train_step_graph(test_images, generated_images, gt_test_images, losses_terms, D_losses_terms,
                     examples_suptitle_text='', losses_suptitle_text='', losses_smooth_window=25):
    num_examples = len(generated_images)
    
    fig, axs = plt.subplots(3, num_examples, figsize=(num_examples*4, 12), squeeze=False)
    for i in range(num_examples):
        axs[0, i].imshow(test_images[i])
        axs[0, i].set_title('Input Image')
        axs[0, i].axis('off')
        axs[1, i].imshow(generated_images[i])
        axs[1, i].set_title('Generated Image')
        axs[1, i].axis('off')
        axs[2, i].imshow(gt_test_images[i])
        axs[2, i].set_title('Ground Truth')
        axs[2, i].axis('off')
    plt.suptitle(examples_suptitle_text)
    fig.tight_layout(pad=2)
    plt.show()

    G_terms = np.array(losses_terms).T
    D_terms = np.array(D_losses_terms).T

    fig, axs = plt.subplots(4, 2, figsize=(6, 3*4), squeeze=False)
    labels = ['valid_l1', 'hole_l1', 'perceptual_pred', 'perceptual_comp', 'style_pred', 'style_comp', 'tv', 'adversarial_loss']
    for i, ax in enumerate([axis for axis in axs.ravel()]):
        ax.plot(smooth1d(G_terms[i], losses_smooth_window), label=labels[i])
        ax.legend()

    plt.suptitle(losses_suptitle_text)
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(6, 3*1), squeeze=False)
    labels = ['real_loss', 'fake_loss']
    for i, ax in enumerate([axis for axis in axs.ravel()]):
        ax.plot(smooth1d(D_terms[i], losses_smooth_window), label=labels[i])
        ax.legend()

    plt.show()

    
def train(model, optimizer, discriminator, discriminator_optimizer, 
          dataloader, validation_dataset, criterion, discriminator_criterion, dataset_mean, dataset_std,  
          epochs=1, graph_show_interval=10, losses_smooth_window=25, device='cpu',
          trained_iters=0, save_interval=10000, save_folder='.', save_name='baseline', discriminator_loss_threshold=0.5):

    batch_size = dataloader.batch_size
    
    inp_test1, mask_test1, target_test1 = validation_dataset[0]
    inp_test2, mask_test2, target_test2 = validation_dataset[1]
    inp_test = torch.stack([inp_test1, inp_test2])
    mask_test = torch.stack([mask_test1, mask_test2])
    target_test = torch.stack([target_test1, target_test2])
    test_images = [tensor2image(im, dataset_mean, dataset_std) for im in inp_test]
    gt_test_images = [tensor2image(im, dataset_mean, dataset_std) for im in target_test]


    losses_storage = []
    loss_terms_storage = []
    discriminator_losses_storage = []
    discriminator_loss_terms_storage = []
    discriminator_prev_loss = discriminator_loss_threshold+1
    for epoch in range(epochs):
        for batch_num, (input, mask, target) in enumerate(dataloader):
            input = input.to(device)
            mask = mask.to(device)
            target = target.to(device)

            model.train()
            discriminator.eval()
            output, _ = model(input, mask)
            fake_probas = discriminator(output, mask[:, 0:1, :, :])
            
            loss_terms = criterion(output, mask, target, fake_probas, separate=True)
            train_step(optimizer, loss_terms, losses_storage, loss_terms_storage)
            
        
            model.eval()
            discriminator.train()
            with torch.no_grad():
                output, _ = model(input, mask)
            fake_probas = discriminator(output, mask[:, 0:1, :, :])
            true_probas = discriminator(target, mask[:, 0:1, :, :])
            discriminator_loss_terms = discriminator_criterion(fake_probas, true_probas, separate=True)
            
            if discriminator_prev_loss > discriminator_loss_threshold:
                train_step(discriminator_optimizer, discriminator_loss_terms, discriminator_losses_storage, discriminator_loss_terms_storage)
            else:
                train_step(None, discriminator_loss_terms, discriminator_losses_storage, discriminator_loss_terms_storage)
            discriminator_prev_loss = discriminator_losses_storage[-1]
            
            trained_iters += input.shape[0]

            losses_save_interval = batch_size
            if (trained_iters % save_interval) < batch_size:
                save_state(save_folder, f'{save_name}_{trained_iters}', model, optimizer, trained_iters, loss_terms_storage, losses_save_interval)
                save_state(save_folder, f'D_{save_name}_{trained_iters}', discriminator, discriminator_optimizer, trained_iters, discriminator_loss_terms_storage, losses_save_interval)


            # Example images and losses graph 
            # -----------------------------------
            if batch_num % graph_show_interval == 0:
                examples_suptitle_text = f"{batch_num+1}/{len(dataloader)}"
                losses_suptitle_text = f"G loss: {losses_storage[-1]}, D loss: {discriminator_losses_storage[-1]}"

                model.eval()
                with torch.no_grad():
                    genims, _ = model(inp_test.to(device), mask_test.to(device))
                    generated_images = [tensor2image(genim, dataset_mean, dataset_std) for genim in genims]
                
                clear_output(wait=True)
                train_step_graph(
                    test_images, generated_images, gt_test_images, 
                    loss_terms_storage, discriminator_loss_terms_storage, 
                    examples_suptitle_text, losses_suptitle_text, losses_smooth_window
                )
            # -----------------------------------
