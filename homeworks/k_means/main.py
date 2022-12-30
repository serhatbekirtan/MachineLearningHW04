if __name__ == "__main__":
    from homeworks.k_means.k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1) # Done
def main():
    """Main function of k-means problem

    You should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """

    (x_train, _), (x_test, _) = load_dataset("mnist")

    # Run Lloyd's Algorithm for k=10
    centroids, classifications, loss = lloyd_algorithm(x_train, 10)

    # Plot of objective function versus iteration
    #plt.plot(loss)
    #plt.xlabel("Iteration")
    #plt.ylabel("Objective function")
    #plt.show()

    centroids_img = []
    for i in centroids:
        centroids_img.append(i.reshape(28, 28))

    # Plot the centroids for k = 10 in a 1x10 grid
    fig, axes = plt.subplots(1, 10)
    for i, (ax, centroid) in enumerate(zip(axes.ravel(), centroids_img)):
        ax.imshow(centroid)
        ax.axis('off')



    ks = [2, 4, 8, 16, 32, 64]
    # Initialize lists to store the objective function values for training and test sets
    train_objs = []
    test_objs = []
    for k in ks:
        # Run Lloyd's Algorithm on the training set
        centroids, classifications, loss = lloyd_algorithm(x_train, k)
        # Append the losses to the lists
        train_objs.append(loss)
        # Calculate the objective function value on the test set
        test_loss = calculate_error(x_test, centroids)
        # Append the losses to the lists
        test_objs.append(test_loss)

        # Print the loss for each k
        print("k = ", k)
        print("Train Loss: ", loss)
        print("Test Loss: ", test_loss)



    # Plot the loss for each k
    plt.plot(ks, train_objs, label="Train Loss")
    plt.plot(ks, test_objs, label="Test Loss")
    plt.legend()
    plt.xlabel("K")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
