import matplotlib.pyplot as plt


def plot_triplet(triplet):
    fig, axs = plt.subplots(1, 3)
    # transform = transforms.ToPILImage()
    # axs[0].imshow(triplet[0])
    axs[0].imshow(triplet[0].permute(1, 2, 0))
    axs[0].set_title("Anchor")

    # axs[1].imshow(triplet[1])
    axs[1].imshow(triplet[1].permute(1, 2, 0))
    axs[1].set_title("Positive")

    # axs[2].imshow(triplet[2])
    axs[2].imshow(triplet[2].permute(1, 2, 0))
    axs[2].set_title("Negative")

    plt.show()
