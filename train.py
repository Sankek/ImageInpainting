import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output


def train_step(optimizer, loss_terms, losses_storage, loss_terms_storage):
    optimizer.zero_grad()
    loss = sum(loss_terms)
    loss_terms = [term.item() for term in loss_terms]
    loss_terms_storage.append(loss_terms)
    losses_storage.append(loss.item())
    loss.backward()
    optimizer.step()

    
def train_step_graph(test_images, generated_images, gt_test_images, losses, D_losses
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

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(smooth1d(np.array(losses), losses_smooth_window), label='G_losses')
    ax.plot(smooth1d(np.array(D_losses), losses_smooth_window), label='D_losses')
    ax.legend()
    plt.suptitle(losses_suptitle_text)
    plt.show()

    
def train(model, optimizer, discriminator, discriminator_optimizer, 
          dataloader, criterion, discriminator_criterion, dataset_mean, dataset_std,  
          epochs=1, graph_show_interval=10, losses_smooth_window=25, device='cpu',
          trained_iters=0, save_interval=10000, save_folder='.', save_name='baseline'):
    
    inp_test1, mask_test1, target_test1 = dataset[0]
    inp_test2, mask_test2, target_test2 = dataset[1]
    # inp_test = inp_test.unsqueeze(0)
    # mask_test = mask_test.unsqueeze(0)
    # target_test = target_test.unsqueeze(0)
    inp_test = torch.stack([inp_test1, inp_test2])
    mask_test = torch.stack([mask_test1, mask_test2])
    target_test = torch.stack([target_test1, target_test2])
    test_images = [tensor2image(im, dataset_mean, dataset_std) for im in inp_test]
    gt_test_images = [tensor2image(im, dataset_mean, dataset_std) for im in target_test]


    losses = []
    loss_terms = []
    discriminator_losses = []
    discriminator_loss_terms = []
    for epoch in range(epochs):
        for batch_num, (input, mask, target) in enumerate(dataloader):
            model.train()
            discriminator.train()
    
            input = input.to(device)
            mask = mask.to(device)
            target = target.to(device)
            
            output, _ = model(input, mask)
            fake_probas = discriminator(output, mask)
            true_probas = discriminator(target, mask)
            
            loss_terms = criterion(output, mask, target, fake_probas, separate=True)
            train_step(optimizer, loss_terms, losses, loss_terms)
            
            discriminator_loss_terms = discriminator_criterion(fake_probas, true_probas, separate=True)
            train_step(discriminator_optimizer, discriminator_loss_terms, discriminator_losses, discriminator_loss_terms)
            
            trained_iters += input.shape[0]

            losses_save_interval = batch_size
            if (trained_iters % save_interval) < batch_size:
                save_state(save_folder, f'{save_name}_{trained_iters}', model, optimizer, trained_iters, loss_terms, losses_save_interval)
                save_state(save_folder, f'D_{save_name}_{trained_iters}', discriminator, discriminator_optimizer, trained_iters, discriminator_loss_terms, losses_save_interval)


            # Example images and losses graph 
            # -----------------------------------
            if batch_num % graph_show_interval == 0:
                losses_suptitle_text = f"{batch_num+1}/{len(dataloader)}"
                examples_suptitle_text = f"G loss: {losses[-1]}, D loss: {discriminator_losses[-1]}"

                model.eval()
                with torch.no_grad():
                    genims, _ = model(inp_test.to(device), mask_test.to(device))
                    generated_images = [tensor2image(genim, dataset_mean, dataset_std) for genim in genims]
                
                clear_output(wait=True)
                train_step_graph(
                    test_images, generated_images, gt_test_images, 
                    losses, discriminator_losses, losses_suptitle_text, losses_smooth_window
                )
            # -----------------------------------
