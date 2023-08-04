import matplotlib.pyplot as plt


def print_samples(loader, count=16):
    """Print the sample input images"""

    fig = plt.figure(figsize=(30, 30))
    for images, labels in loader:
        for i in range(count):
            axis = fig.add_subplot(4, 4, i+1)
            axis.set_title(f'Label: {labels[i]}')
            plt.imshow(images[i].numpy().transpose(1, 2, 0))
        break


def plot_incorrect_predictions(predictions, class_map, title="Incorrect Predictions", count=10):
    """ Plots Incorrect predictions """
    print(f'Incorrect Predictions (Target/Predicted)')

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    fig = plt.figure(figsize=(10, 5))
    plt.title(title)
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{class_map[t]}/{class_map[p]}')
        plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break
